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

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

# -------------------------
# Parameters
# -------------------------
Ri, Ro = 1.0, 8.0
Nphi, Nr = 512, 512
Re_target = 40.0   # or 60.0
Rref = 1.0      # Chosen reference Radius
Omega_ref = 2*np.pi # Chosen reference frequency
nu = (Rref**2) *  Omega_ref / Re_target    # since Rref=1


dealias = 3/2
dtype = np.float64

stop_sim_time = 10.0
timestepper = d3.SBDF2
max_dt = 1e-4
# -------------------------
# Bases / distributor
# -------------------------
coords = d3.PolarCoordinates('phi', 'r')
dist   = d3.Distributor(coords, dtype=dtype)

annulus = d3.AnnulusBasis(
    coords,
    shape=(Nphi, Nr),
    radii=(Ri, Ro),
    dealias=(dealias, dealias),
    dtype=dtype,
)

# phi-only basis container (OK to use inner_edge or outer_edge)
edge = annulus.outer_edge

# -------------------------
# Fields
# -------------------------
p = dist.Field(name='p', bases=annulus)
u = dist.VectorField(coords, name='u', bases=annulus)

# tau fields live on the phi-only basis
tau_p  = dist.Field(name='tau_p', bases=edge)                     # scalar (phi-only)
tau_u1 = dist.VectorField(coords, name='tau_u1', bases=edge)      # vector (phi-only)
tau_u2 = dist.VectorField(coords, name='tau_u2', bases=edge)      # vector (phi-only)

# prescribed outer boundary velocity (phi-only)
u_outer = dist.VectorField(coords, name='u_outer', bases=edge)

# -------------------------
# Grids for BC construction
# -------------------------
phi, r = dist.local_grids(annulus, scales=dealias)
(phi_e,) = dist.local_grids(edge, scales=dealias)
if phi_e.ndim == 1:
    phi_e = phi_e[:, None]

# -------------------------
# Tau lifting (radial direction)
# -------------------------
# lift basis: prefer radial_basis().derivative_basis(1) when available
rb = annulus.radial_basis() if callable(getattr(annulus, "radial_basis", None)) else annulus.radial_basis
lift_basis = rb.derivative_basis(1) if hasattr(rb, "derivative_basis") else annulus.derivative_basis(1)
lift = lambda A: d3.Lift(A, lift_basis, -1)

# unit radial direction e_r (component order is [phi, r])
er = dist.VectorField(coords, name='er', bases=annulus)
er.change_scales(dealias)
er['g'][0] = 0.0
er['g'][1] = 1.0
er.change_scales(1)

# first-order reduction
grad_u = d3.grad(u) + er * lift(tau_u1)

divu  = d3.div(u)
omega = -d3.div(d3.skew(u))


area = np.pi * (Ro**2 - Ri**2)          # area of 2D annulus with phi in [0,2pi)
div2_int = d3.integ(divu*divu)          # operators you can evaluate later
om2_int  = d3.integ(omega*omega)

# -------------------------
# Outer BC forcing (translation in x)
# -------------------------
Amp = 0.2  # set to 0.0 for the “should stay at rest” test

def U_of_t(t):
    return Amp * np.pi * (np.cos(2*np.pi*t) + 2*np.cos(4*np.pi*t))

def ramp(t, t_ramp=0.05):
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
CFL = d3.CFL(solver, initial_dt=max_dt, cadence=1, safety=0.5,
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
        dt = CFL.compute_timestep()

        # Set BC at the time we're stepping to
        t_bc = solver.sim_time + dt
        set_outer_bc(t_bc)

        solver.step(dt)

        # One-time BC check (after step so sim_time is the new time)
        if solver.iteration == 10:
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
        if solver.iteration % 10 == 0:
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

    flux_outer = Ro * np.trapz(ur_o, x=phi_line)

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