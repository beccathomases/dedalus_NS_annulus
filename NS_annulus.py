"""
NS_annulus.py

2D incompressible Navier–Stokes in an annulus using Dedalus v3 (d3).

Geometry / coordinates:
  - PolarCoordinates('phi','r') on an annulus r ∈ [Ri, Ro], phi ∈ [0, 2π)
  - AnnulusBasis: Fourier in phi and a radial spectral basis on [Ri, Ro]

Equations (incompressible NS):
  - div(u) = 0
  - dt(u) + u·grad(u) = -grad(p) + nu * Laplacian(u)

Numerics / formulation:
  - Implemented as an IVP with Dedalus' two-boundary tau formulation
    (tau_p, tau_u1, tau_u2).
  - Viscous term is written in first-order form with tau lifting:
        grad_u = grad(u) + rvec * Lift(tau_u1, ...)

Boundary conditions (current configuration):
  - Inner wall r = Ri: no-slip, u = 0
  - Outer wall r = Ro: prescribed velocity u = u_outer
    Current choice is steady Couette rotation with angular speed Omega_out:
      u_phi(Ro) = Omega_out * Ro,  u_r(Ro) = 0

Initial condition (current configuration):
  - Exact laminar Couette profile (phi-independent), consistent with the BCs.
  - With this IC + BC choice, the script serves as a quick correctness check.

Outputs:
  - Writes HDF5 snapshots to ./snapshots/ (u, p, vorticity, divu)
  - Logs diagnostics every 50 iterations:
      max|u|, rms(div), rms(vorticity)
  - End-of-run check:
      relative L2 error of the phi-averaged u_phi(r) vs. analytic Couette profile.

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
Ri, Ro = 1.0, 2.0
Nphi, Nr = 128, 128
Re = 200.0
nu = 1.0 / Re
Omega_out = 1.0   # angular velocity of outer cylinder

dealias = 3/2
dtype = np.float64

stop_sim_time = 10.0
timestepper = d3.SBDF2
max_dt = 1e-2

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
# Boundary forcing you control
# -------------------------
def set_outer_bc(t):
    Uo = Omega_out * Ro

    u_outer.change_scales(dealias)
    u_outer['g'][0] = Uo   # azimuthal component u_phi at r=Ro
    u_outer['g'][1] = 0.0  # radial component u_r at r=Ro

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
# Initial condition: exact Couette profile (phi-independent)
# -------------------------

rline = r[0, :] if r.ndim == 2 else r
uphi_exact = Omega_out * (Ro**2/(Ro**2 - Ri**2)) * (rline - (Ri**2)/rline)  # u_phi(r)

u.change_scales(dealias)
u['g'][0] = uphi_exact[None, :]   # uses same component you use in u_outer['g'][0]
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
        # update boundary data (safe to call every step)
        set_outer_bc(solver.sim_time)

        dt = CFL.compute_timestep()
        solver.step(dt)

        # one-time BC check (after first step so u exists on the grid)
        if solver.iteration == 10:
            u.change_scales(dealias)
            Uo = Omega_out * Ro
            logger.info(
                f"BC check: max|u_phi(Ro)-Uo| = {np.max(np.abs(u['g'][0][:,-1] - Uo)):.3e}, "
                f"max|u_r(Ro)| = {np.max(np.abs(u['g'][1][:,-1])):.3e}"
            )
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

    # --- compare mean u_phi(r) with analytic Couette profile at final time ---
    u.change_scales(dealias)
    uphi = u['g'][0]                   # azimuthal component, shape (Nphi, Nr)
    uphi_mean = np.mean(uphi, axis=0)  # average over phi -> (Nr,)

    rline = r[0, :] if r.ndim == 2 else r
    uphi_exact = Omega_out * (Ro**2/(Ro**2 - Ri**2)) * (rline - (Ri**2)/rline)

    rel_err = np.linalg.norm(uphi_mean - uphi_exact) / np.linalg.norm(uphi_exact)
    logger.info(f"Final Couette profile relative L2 error (phi-avg): {rel_err:.3e}")

finally:
    if hasattr(solver, "start_time_end"):
        solver.log_stats()