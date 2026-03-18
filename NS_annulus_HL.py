"""
NS_annulus.py

2D incompressible Navier-Stokes in an annulus using Dedalus v3 (d3).

Geometry / coordinates:
  - PolarCoordinates('phi','r') on an annulus r in [Ri, Ro], phi in [0, 2pi)
  - AnnulusBasis: Fourier in phi and a radial spectral basis on [Ri, Ro]

Equations (incompressible NS):
  - div(u) = 0
  - dt(u) + u dot grad(u) = -grad(p) + nu * Laplacian(u)

Numerics / formulation:
  - Implemented as an IVP with Dedalus' two-boundary tau formulation
    (tau_p, tau_u1, tau_u2).
  - Viscous term is written in first-order form with tau lifting:
        grad_u = grad(u) + rvec * Lift(tau_u1, ...)

Boundary conditions (current configuration):
  - Inner wall r = Ri: no-slip, u = 0.
  - Outer wall r = Ro: prescribed oscillatory horizontal velocity (position fixed):
        u_x(t) = Amp * pi * [cos(2*pi*t) + 2*cos(4*pi*t)],   u_y(t) = 0.
    Implemented in polar components as:
        u_r(phi,t)   = u_x(t) cos(phi)
        u_phi(phi,t)   = -u_x(t) sin(phi)

Reynolds number convention:
  - Re = R^2 * Omega / nu with R = 1 and Omega = 2*pi, so nu = 2*pi / Re.

Initial condition (current configuration):  
  - Start from rest (u = 0), then apply the time-dependent outer boundary forcing.

Outputs:
  - Writes HDF5 snapshots to ./snapshots/ (u, p, vorticity, divu)
  - Logs diagnostics every 50 iterations:
      max|u|, rms(div), rms(vorticity)
 
Notes:
  - Component ordering for PolarCoordinates('phi','r'):
      u['g'][0] = azimuthal component (u_phi), u['g'][1] = radial component (u_r)
  - set_outer_bc(t) is the intended hook for changing the outer-wall forcing.
"""
import numpy as np
import dedalus.public as d3
import logging
import h5py
from scipy.interpolate import RegularGridInterpolator
from collections import deque
from mpi4py import MPI

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

comm = MPI.COMM_WORLD
rank = comm.rank
nproc = comm.size
size = comm.size

# -------------------------
# Parameters
# -------------------------
Ri, Ro = 1.0, 8.0
Nphi, Nr = 128, 128
Re_target = 5   # or 60.0
Rref = 1.0      # Chosen reference Radius
Omega_ref = 2*np.pi # Chosen reference frequency
nu = (Rref**2) *  Omega_ref / Re_target    # since Rref=1

Amp = 0.2  # set to 0.0 for the “should stay at rest” test

dealias = 3/2
dtype = np.float64

stop_sim_time = 0.01
timestepper = d3.SBDF2
max_dt = 1e-4
bc_time_offset = 0.0

# -------------------------
# Bases / distributor
# -------------------------
coords = d3.PolarCoordinates('phi', 'r')
dist   = d3.Distributor(coords, dtype=dtype)

annulus = d3.AnnulusBasis(
    coords,
    shape=(Nphi, Nr),
    radii=(Ri, Ro),
    dealias=(dealias, dealias),   # Annulus wants a 2-tuple
    dtype=dtype
)

# A 1D "edge" basis for tau fields / boundary data (phi-only).
# Annulus has inner_edge and outer_edge; either works as a phi-only basis container.
edge = annulus.outer_edge

# -------------------------
# Fields
# -------------------------
p = dist.Field(name='p', bases=annulus)
u = dist.VectorField(coords, name='u', bases=annulus)

# Tau terms (2 boundaries => 2 tau terms for velocity; plus tau for divergence/pressure constraint)
tau_p  = dist.Field(name='tau_p')
tau_u1 = dist.VectorField(coords, name='tau_u1', bases=edge)
tau_u2 = dist.VectorField(coords, name='tau_u2', bases=edge)

# Prescribed outer boundary velocity (a known field you update in time)
u_outer = dist.VectorField(coords, name='u_outer', bases=edge)

# Grids
phi, r = dist.local_grids(annulus, scales=dealias)

# phi_e is currently unused; kept for future phi-dependent BCs.
(phi_e,) = dist.local_grids(edge, scales=dealias)
if phi_e.ndim == 1:
    phi_e = phi_e[:, None]

# --- Unit radial vector / r-vector for tau lifting (annulus) ---
rbasis = annulus.radial_basis              # NOTE: no parentheses

# Use the annulus radial grid and collapse away phi (keep shape (1, Nr_dealias))
if r.ndim == 1:
    r1d = r[None, :]
else:
    r1d = r[0:1, :]   # take one phi row -> (1, Nr_dealias)

rvec = dist.VectorField(coords, name='rvec', bases=rbasis)   # position-like radial vector (0, r)

rvec.change_scales(dealias)

# Components: [0] = azimuthal, [1] = radial in PolarCoordinates('phi','r')

rvec['g'][0] = 0.0
rvec['g'][1] = r1d

# Tau lift helper
lift_basis = annulus.derivative_basis(1)
lift = lambda A: d3.Lift(A, lift_basis, -1)

# First-order reduction for viscous term (tau enters via rvec * lift(tau_u1))
grad_u = d3.grad(u) + rvec * lift(tau_u1)

divu = d3.div(u)
omega = -d3.div(d3.skew(u))   # scalar vorticity (out of plane)

area = np.pi * (Ro**2 - Ri**2)          # area of 2D annulus with phi in [0,2pi)
div2_int = d3.integ(divu*divu)          # operators you can evaluate later
om2_int  = d3.integ(omega*omega)


# -------------------------
# Outer BC forcing (translation in x)
# -------------------------


def U_of_t(t):
    return Amp * np.pi * (np.cos(2*np.pi*t) + 2*np.cos(4*np.pi*t))

def ramp(t, t_ramp=0.5):
    return 1 - np.exp(-t/t_ramp)

def set_outer_bc(t):
    U = ramp(t) * U_of_t(t)
    u_outer.change_scales(dealias)
    u_outer['g'][0] = -U * np.sin(phi_e)   # u_phi
    u_outer['g'][1] =  U * np.cos(phi_e)   # u_r
    u_outer.change_scales(1)               # good habit: keep coeff scale consistent


# -------------------------
# Problem
# -------------------------
problem = d3.IVP([p, u, tau_p, tau_u1, tau_u2], namespace=locals())
problem.add_equation("trace(grad_u) + tau_p = 0")
problem.add_equation("dt(u) - nu*div(grad_u) + grad(p) + lift(tau_u2) = - u@grad(u)")

# BCs: inner no-slip, outer prescribed velocity
problem.add_equation("u(r=Ri) = 0")
problem.add_equation("u(r=Ro) = u_outer")

# Pressure gauge (remove nullspace)
problem.add_equation("integ(p) = 0")

solver = problem.build_solver(timestepper)
solver.stop_sim_time = stop_sim_time

snapshots = solver.evaluator.add_file_handler('snapshots', sim_dt=0.1, max_writes=200)
snapshots.add_task(u, name='u')
snapshots.add_task(p, name='p')
snapshots.add_task(omega, name='vorticity')
snapshots.add_task(divu, name='divu')


# -------------------------
# Initial condition: start from rest
# -------------------------

u.change_scales(dealias)
u['g'][0] = 0.0
u['g'][1] = 0.0
u.change_scales(1)

# -------------------------
# CFL (optional but recommended)
# -------------------------
CFL = d3.CFL(solver, initial_dt=max_dt, cadence=10, safety=0.8,
             max_change=1.5, min_change=0.5, max_dt=max_dt)
CFL.add_velocity(u)


flow = d3.GlobalFlowProperty(solver, cadence=20)
flow.add_property(u@u, name='u2')
flow.add_property(divu*divu, name='div2')
flow.add_property(omega*omega, name='om2')

# -------------------------
# Main loop
# -------------------------
logger.info("Starting loop")
try:
    while solver.proceed:
        print(f"[rank {rank}] top of loop, iter={solver.iteration}, t={solver.sim_time}", flush=True)

        dt = max_dt
        print(f"[rank {rank}] got fixed dt={dt}", flush=True)

        t_bc = solver.sim_time + dt + bc_time_offset
        print(f"[rank {rank}] before set_outer_bc, t_bc={t_bc}", flush=True)

        set_outer_bc(t_bc)
        print(f"[rank {rank}] after set_outer_bc", flush=True)

        solver.step(dt)
        print(f"[rank {rank}] finished step", flush=True)

        print(f"[rank {rank}] after step, iter={solver.iteration}, t={solver.sim_time}", flush=True)

        t_phys = solver.sim_time + bc_time_offset
        print(f"[rank {rank}] before maybe_store_field_history", flush=True)
        maybe_store_field_history(t_phys)
        print(f"[rank {rank}] after maybe_store_field_history", flush=True)

        # One-time BC check (after step so sim_time is the new time)
        if solver.iteration == 10:
            print(f"[rank {rank}] entering diag block", flush=True)

            print(f"[rank {rank}] before flow.max", flush=True)
            max_u = np.sqrt(flow.max('u2'))
            print(f"[rank {rank}] after flow.max", flush=True)

            print(f"[rank {rank}] before div2 evaluate", flush=True)
            div2_val = div2_int.evaluate()['g'].item()
            print(f"[rank {rank}] after div2 evaluate", flush=True)

            print(f"[rank {rank}] before om2 evaluate", flush=True)
            om2_val  = om2_int.evaluate()['g'].item()
            print(f"[rank {rank}] after om2 evaluate", flush=True)
            u.change_scales(dealias)
            t_now = solver.sim_time
            U = ramp(t_now) * U_of_t(t_now)

            phi_line = phi[:, 0] if phi.ndim == 2 else phi
            ur  = u['g'][1][:, -1]
            uph = u['g'][0][:, -1]

            ux = ur*np.cos(phi_line) - uph*np.sin(phi_line)
            uy = ur*np.sin(phi_line) + uph*np.cos(phi_line)

            logger.info(
                f"BC check: max|u_x(Ro)-U|={np.max(np.abs(ux-U)):.3e}, "
                f"max|u_y(Ro)|={np.max(np.abs(uy)):.3e}"
            )
            u.change_scales(1)

        # Stop early if solution goes non-finite (check occasionally)
        if solver.iteration % 500 == 0:
            u.change_scales(dealias)
            if not np.isfinite(u['g']).all():
                logger.error("Non-finite u detected; stopping.")
                break
            u.change_scales(1)

        # periodic diagnostics
        if (solver.iteration - 1) % 50 == 0:
            max_u = np.sqrt(flow.max('u2'))
            div2_val = div2_int.evaluate()['g'].item()
            om2_val  = om2_int.evaluate()['g'].item()
            rms_div  = np.sqrt(div2_val / area)
            rms_om   = np.sqrt(om2_val  / area)
            logger.info(
                f"iter={solver.iteration:6d} t={solver.sim_time:9.4f} "
                f"dt={dt:9.2e} max|u|={max_u:9.2e} rms(div)={rms_div:9.2e} rms(ω)={rms_om:9.2e}"
            )
            # -------------------------
    # End-of-run summaries
    # -------------------------
    u.change_scales(dealias)

    # grid arrays (same scales)
    phi_g, r_g = dist.local_grids(annulus, scales=dealias)
    phi_line = phi_g[:, 0] if phi_g.ndim == 2 else phi_g
    rline    = r_g[0, :]   if r_g.ndim == 2 else r_g

    ur  = u['g'][1]
    uph = u['g'][0]

    # Build u_x everywhere (Cartesian)
    ux = ur*np.cos(phi_g) - uph*np.sin(phi_g)

    # Fourier in phi: take m=1 coefficient vs r
    ux_hat = np.fft.rfft(ux, axis=0) / ux.shape[0]
    m1 = ux_hat[1, :]
    amp  = np.abs(m1)
    phase = np.angle(m1)

    np.savez("compare_m1_ux_final.npz", r=rline, amp=amp, phase=phase, Re=Re_target, Ro=Ro)

    # Some global scalars
    div2_val = div2_int.evaluate()['g'].item()
    om2_val  = om2_int.evaluate()['g'].item()
    rms_div  = np.sqrt(div2_val / area)
    rms_om   = np.sqrt(om2_val  / area)

    KE  = 0.5 * d3.integ(u@u).evaluate()['g'].item()
    Ens = 0.5 * d3.integ(omega*omega).evaluate()['g'].item()

    # Boundary checks at final time (optional)
    t_end = solver.sim_time
    U_end = ramp(t_end) * U_of_t(t_end)

    ur_o  = u['g'][1][:, -1]
    uph_o = u['g'][0][:, -1]
    ux_o = ur_o*np.cos(phi_line) - uph_o*np.sin(phi_line)
    uy_o = ur_o*np.sin(phi_line) + uph_o*np.cos(phi_line)

    bc_x_err = np.max(np.abs(ux_o - U_end))
    bc_y_err = np.max(np.abs(uy_o))

    ur_i  = u['g'][1][:, 0]
    uph_i = u['g'][0][:, 0]
    inner_slip = np.max(np.sqrt(ur_i**2 + uph_i**2))

    flux_outer = Ro * 2*np.pi * np.mean(ur_o)

    logger.info(
        "END SUMMARY: "
        f"t={t_end:.6f}  "
        f"max|u_x(Ro)-U|={bc_x_err:.3e}  max|u_y(Ro)|={bc_y_err:.3e}  "
        f"max|u(Ri)|={inner_slip:.3e}  "
        f"rms(div)={rms_div:.3e}  rms(ω)={rms_om:.3e}  "
        f"KE={KE:.6e}  Ens={Ens:.6e}  "
        f"flux_outer={flux_outer:.3e}"
    )

finally:
    if hasattr(solver, "start_time_end"):
        solver.log_stats()