"""
NS_annulus.py

2D incompressible Navier--Stokes in an annulus using Dedalus v3.

Geometry / coordinates
----------------------
- Polar coordinates (phi, r) on an annulus with
      r in [Ri, Ro],   phi in [0, 2*pi)
- AnnulusBasis: Fourier in phi and radial spectral basis in r

Equations
---------
We solve the incompressible Navier--Stokes equations
    div(u) = 0
    dt(u) + u · grad(u) = -grad(p) + nu * div(grad_u)

using Dedalus' tau formulation on the annulus.

Numerical formulation
---------------------
- IVP unknowns:
      p, u, tau_p, tau_u1, tau_u2
- First-order viscous form with tau lifting:
      grad_u = grad(u) + rvec * Lift(tau_u1)
- Pressure gauge condition:
      integ(p) = 0

Boundary conditions
-------------------
- Inner wall r = Ri:
      u = 0
- Outer wall r = Ro:
      prescribed oscillatory horizontal translation
      u_x(t) = Amp * pi * [cos(2*pi*t) + 2*cos(4*pi*t)]
      u_y(t) = 0

In polar components this is imposed as
      u_r(phi,t)   =  u_x(t) cos(phi)
      u_phi(phi,t) = -u_x(t) sin(phi)

Reynolds number convention
--------------------------
Using reference radius Rref = 1 and reference frequency Omega_ref = 2*pi,
the viscosity is
      nu = Rref^2 * Omega_ref / Re_target.

Run organization
----------------
Runs are organized by
      case_label/segment_label

Examples:
- seg01_from_rest_snap0p1
- seg02_continue_snap0p01

Each segment writes into its own folder:
      run_dir/
          snapshots/
          checkpoints/
          run_info.txt

Restart modes
-------------
- restart_mode = "none"
      start from rest
- restart_mode = "full"
      exact continuation from a checkpoint file

Time forcing / ramp
-------------------
- A startup ramp may be used for fresh runs from rest.
- By default, the ramp is disabled automatically for restart runs so that
  continuation uses the physical boundary forcing directly.

Outputs
-------
- Snapshots:
      u, p, vorticity, divu
  written every snapshot_dt to run_dir/snapshots
- Checkpoints:
      full solver state
  written to run_dir/checkpoints for restart
- run_info.txt:
      records parameters and restart metadata for the segment

Runtime guardrails
------------------
- early boundary-condition sanity check
- periodic NaN/Inf check on main fields
- minimal progress logging (iteration, time, dt)
- if non-finite values are detected, the run stops early and skips final
  summary diagnostics

Notes
-----
- Component ordering for PolarCoordinates('phi','r'):
      u['g'][0] = u_phi
      u['g'][1] = u_r
- In MPI mode, detailed diagnostics are intended to be done in postprocessing.
"""

import numpy as np
import dedalus.public as d3
import logging
from mpi4py import MPI
import os
from pathlib import Path
import re

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

comm = MPI.COMM_WORLD
rank = comm.rank
size = comm.size

print(f"MPI hello from rank {rank} of {size}", flush=True)

# ============================================================
# User-set run parameters
# ============================================================
base_run_root = Path("/work/pi_bthomases_smith_edu/bthomases_smith_edu/runs/dedalus_NS_annulus")

Re_target = 50.0
Amp       = 0.1

t0 = 10.0
t1 = 100.0

restart_index = -1
manual_restart_file = ""   # optional explicit override

checkpoint_dt = 1.0
snapshot_dt   = 50.0
max_dt        = 1e-4

Ri, Ro = 1.0, 8.0
Nphi, Nr = 256, 512


# ============================================================
# Naming helpers
# ============================================================
SET_RE = re.compile(r"_s(\d+)(?:\.h5)?$")

def amp_tag(a, ndp=8):
    s = f"{a:.{ndp}f}".rstrip("0").rstrip(".")
    if s.startswith("0."):
        s = "p" + s[2:]
    else:
        s = s.replace(".", "p")
    s = s.replace("-", "m")
    return s

def time_tag(t, width=8, ndp=2):
    return f"{t:0{width}.{ndp}f}".replace(".", "p")

def case_name(Re, amp):
    return f"Re{int(Re):d}_A{amp_tag(amp)}"

def segment_name(t0, t1):
    return f"seg_t{time_tag(t0)}_to_t{time_tag(t1)}"

def set_num(path_obj):
    m = SET_RE.search(path_obj.name)
    return int(m.group(1)) if m else -1

def find_previous_segment_dir(case_dir, t0):
    """
    Find the most recent segment directory in this case whose end time is t0.
    Supports 0->10 followed by 10->100, then 100->200, etc.
    """
    case_dir = Path(case_dir)
    endtag = time_tag(t0)
    matches = [
        p for p in case_dir.iterdir()
        if p.is_dir() and p.name.endswith(f"_to_t{endtag}")
    ]
    if not matches:
        raise FileNotFoundError(
            f"No previous segment ending at t={t0} found in {case_dir}"
        )
    matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return matches[0]

def latest_checkpoint_file(checkpoint_root):
    """
    Find the newest checkpoint set in a Dedalus checkpoint folder.
    """
    checkpoint_root = Path(checkpoint_root)
    if not checkpoint_root.exists():
        raise FileNotFoundError(f"Checkpoint folder does not exist: {checkpoint_root}")

    top_files = sorted(checkpoint_root.glob("checkpoints_s*.h5"), key=set_num)
    if top_files:
        return str(top_files[-1])

    set_dirs = sorted(
        [p for p in checkpoint_root.iterdir()
         if p.is_dir() and re.match(r"^checkpoints_s\d+$", p.name)],
        key=set_num
    )
    if not set_dirs:
        raise FileNotFoundError(f"No checkpoint sets found in {checkpoint_root}")

    latest_set = set_dirs[-1]

    joint_inside = latest_set / f"{latest_set.name}.h5"
    if joint_inside.exists():
        return str(joint_inside)

    h5s = sorted(latest_set.glob("*.h5"))
    if not h5s:
        raise FileNotFoundError(f"No .h5 files found inside {latest_set}")

    p0_files = [f for f in h5s if f.name.endswith("_p0.h5")]
    if p0_files:
        return str(p0_files[0])

    h5s.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return str(h5s[0])


# ============================================================
# Derived paths and restart logic
# ============================================================
case_label    = case_name(Re_target, Amp)
segment_label = segment_name(t0, t1)

case_dir = base_run_root / case_label
run_dir  = case_dir / segment_label

prev_segment_dir = None

if manual_restart_file:
    restart_mode = "full"
    restart_file = manual_restart_file
elif abs(t0) < 1e-14:
    restart_mode = "none"
    restart_file = ""
else:
    prev_segment_dir = find_previous_segment_dir(case_dir, t0)
    restart_file = latest_checkpoint_file(prev_segment_dir / "checkpoints")
    restart_mode = "full"

segment_duration = t1 - t0

print("--------------------------------------------------")
print(f"case_label       = {case_label}")
print(f"segment_label    = {segment_label}")
print(f"run_dir          = {run_dir}")
print(f"prev_segment_dir = {prev_segment_dir if prev_segment_dir else '(none)'}")
print(f"restart_mode     = {restart_mode}")
print(f"restart_file     = {restart_file if restart_file else '(none)'}")
print(f"segment t0->t1   = {t0} -> {t1}")
print("--------------------------------------------------")

if rank == 0:
    logger.info(f"run_dir = {run_dir}")
    os.makedirs(run_dir, exist_ok=True)
comm.Barrier()

# Time/forcing controls
bc_time_offset = 0.0
ramp_time = 1.0
use_ramp = (restart_mode == "none")

# -------------------------
# Parameters
# -------------------------
Rref = 1.0
Omega_ref = 2*np.pi
nu = (Rref**2) * Omega_ref / Re_target

dealias = 3/2
dtype = np.float64
timestepper = d3.SBDF2



# -------------------------
# Minimal run-time guardrails
# -------------------------
log_iter = 1000         # print iter, t, dt every 1000 steps
nan_check_iter = 500   # check for NaN/Inf every 500 steps
bc_check_iter = 10     # one early BC sanity check

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

# - 2 pi A (cos(2pi t) + 2 cos(4pi t)),
def U_of_t(t):
    return -2 * Amp * np.pi * (np.cos(2*np.pi*t) + 2*np.cos(4*np.pi*t))

def ramp(t, t_ramp=ramp_time):
    if not use_ramp:
        return 1.0
    return 1.0 - np.exp(-t / t_ramp)

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

# -------------------------
# Restart handling
# -------------------------
file_handler_mode = "overwrite"

if restart_mode == "full":
    write, initial_dt = solver.load_state(restart_file, index=restart_index)

    if abs(solver.sim_time - t0) > max(10*max_dt, 1e-10):
        raise RuntimeError(
            f"Restart landed at sim_time={solver.sim_time:.12f}, expected t0={t0:.12f}"
        )

    phys_start_time = solver.sim_time
    bc_time_offset = 0.0
    solver.stop_sim_time = t1

    logger.info(f"Loaded full state from {restart_file}")
    logger.info(f"Restart sim_time = {solver.sim_time:.12f}")
    logger.info(f"Continuing from physical time {phys_start_time:.6f} to {t1:.6f}")

elif restart_mode == "none":
    u.change_scales(dealias)
    u['g'][0] = 0.0
    u['g'][1] = 0.0
    u.change_scales(1)

    initial_dt = max_dt
    bc_time_offset = 0.0
    phys_start_time = 0.0
    solver.stop_sim_time = t1

    logger.info(f"Starting from rest; physical time will run from 0 to {t1:.6f}")

else:
    raise ValueError(
        f"Unsupported restart_mode = {restart_mode!r}. "
        "Expected 'none' or 'full'."
    )

# -------------------------
# Output handlers
# -------------------------
snapshots = solver.evaluator.add_file_handler(
    os.path.join(run_dir, 'snapshots'),
    sim_dt=snapshot_dt,
    max_writes=200,
    mode=file_handler_mode
)
snapshots.add_task(u, name='u')
snapshots.add_task(p, name='p')
snapshots.add_task(omega, name='vorticity')
snapshots.add_task(divu, name='divu')

checkpoints = solver.evaluator.add_file_handler(
    os.path.join(run_dir, 'checkpoints'),
    sim_dt=checkpoint_dt,
    max_writes=1,
    mode=file_handler_mode
)
checkpoints.add_tasks(solver.state)




if rank == 0:
    with open(os.path.join(run_dir, "run_info.txt"), "w") as fp:
        fp.write(f"case_label = {case_label}\n")
        fp.write(f"segment_label = {segment_label}\n")
        fp.write(f"Re_target = {Re_target}\n")
        fp.write(f"Nphi = {Nphi}\n")
        fp.write(f"Nr = {Nr}\n")
        fp.write(f"Ri = {Ri}\n")
        fp.write(f"Ro = {Ro}\n")
        fp.write(f"nu = {nu:.16e}\n")
        fp.write(f"Amp = {Amp}\n")
        fp.write(f"restart_mode = {restart_mode}\n")
        fp.write(f"restart_file = {restart_file}\n")
        fp.write(f"restart_index = {restart_index}\n")
        fp.write(f"snapshot_dt = {snapshot_dt}\n")
        fp.write(f"checkpoint_dt = {checkpoint_dt}\n")
        fp.write(f"segment_duration = {segment_duration}\n")
        fp.write(f"use_ramp = {use_ramp}\n")
        fp.write(f"ramp_time = {ramp_time}\n")
        fp.write(f"phys_start_time = {phys_start_time:.12f}\n")
        if restart_mode == "full":
            fp.write(f"phys_end_target = {solver.stop_sim_time:.12f}\n")
        else:
            fp.write(f"phys_end_target = {phys_start_time + segment_duration:.12f}\n")




def check_finite_state():
    """
    Check main state/BC fields for NaN/Inf on this rank.
    Returns a list of field names that contain non-finite values locally.
    """
    bad = []

    u.change_scales(1)
    p.change_scales(1)
    u_outer.change_scales(1)
    tau_u1.change_scales(1)
    tau_u2.change_scales(1)

    if not np.isfinite(u['g']).all():
        bad.append("u")
    if not np.isfinite(p['g']).all():
        bad.append("p")
    if not np.isfinite(tau_p['g']).all():
        bad.append("tau_p")
    if not np.isfinite(tau_u1['g']).all():
        bad.append("tau_u1")
    if not np.isfinite(tau_u2['g']).all():
        bad.append("tau_u2")
    if not np.isfinite(u_outer['g']).all():
        bad.append("u_outer")

    return bad

# -------------------------
# Main loop
# -------------------------

stopped_bad = False
logger.info("Starting loop")
try:
    while solver.proceed:
        dt = max_dt
        t_bc = solver.sim_time + dt + bc_time_offset

        set_outer_bc(t_bc)
        solver.step(dt)

        t_phys = solver.sim_time + bc_time_offset

        # One-time BC sanity check
        if solver.iteration == bc_check_iter:
            u.change_scales(dealias)
            t_now = solver.sim_time + bc_time_offset
            U = ramp(t_now) * U_of_t(t_now)

            phi_line = phi[:, 0] if phi.ndim == 2 else phi
            ur  = u['g'][1][:, -1]
            uph = u['g'][0][:, -1]

            ux = ur*np.cos(phi_line) - uph*np.sin(phi_line)
            uy = ur*np.sin(phi_line) + uph*np.cos(phi_line)

            local_bc_x_err = np.max(np.abs(ux - U))
            local_bc_y_err = np.max(np.abs(uy))

            bc_x_err = comm.allreduce(float(local_bc_x_err), op=MPI.MAX)
            bc_y_err = comm.allreduce(float(local_bc_y_err), op=MPI.MAX)

            if rank == 0:
                logger.info(
                    f"BC check: max|u_x(Ro)-U|={bc_x_err:.3e}, "
                    f"max|u_y(Ro)|={bc_y_err:.3e}"
                )

            u.change_scales(1)

        # Stop early if any important field goes non-finite
        if solver.iteration % nan_check_iter == 0:
            bad_local = check_finite_state()
            any_bad = comm.allreduce(int(len(bad_local) > 0), op=MPI.SUM)

            if bad_local:
                logger.error(
                    f"Rank {rank}: non-finite values detected in {bad_local} "
                    f"at iter={solver.iteration}, t={t_phys:.6f}"
                )

            if any_bad > 0:
                stopped_bad = True
                if rank == 0:
                    logger.error("Stopping: non-finite values detected on one or more MPI ranks.")
                break
            

        # Minimal progress log
        if ((solver.iteration - 1) % log_iter == 0) and (rank == 0):
            logger.info(
                f"iter={solver.iteration:6d} "
                f"t={t_phys:9.4f} "
                f"dt={dt:9.2e}"
            )

    # -------------------------
    # End-of-run summaries
    # -------------------------
    if stopped_bad:
        if rank == 0:
            logger.info("Run ended after non-finite detection; skipping final summary diagnostics.")

    elif size == 1:
        u.change_scales(dealias)

        phi_g, r_g = dist.local_grids(annulus, scales=dealias)
        phi_line = phi_g[:, 0] if phi_g.ndim == 2 else phi_g
        rline    = r_g[0, :] if r_g.ndim == 2 else r_g

        ur  = u['g'][1]
        uph = u['g'][0]
        ux = ur*np.cos(phi_g) - uph*np.sin(phi_g)

        ux_hat = np.fft.rfft(ux, axis=0) / ux.shape[0]
        m1 = ux_hat[1, :]
        amp = np.abs(m1)
        phase = np.angle(m1)
        np.savez(os.path.join(run_dir, "compare_m1_ux_final.npz"),
                r=rline, amp=amp, phase=phase, Re=Re_target, Ro=Ro)

        div2_val = div2_int.evaluate()['g'].item()
        om2_val  = om2_int.evaluate()['g'].item()
        rms_div  = np.sqrt(div2_val / area)
        rms_om   = np.sqrt(om2_val  / area)

        KE  = 0.5 * d3.integ(u@u).evaluate()['g'].item()
        Ens = 0.5 * d3.integ(omega*omega).evaluate()['g'].item()

        t_end = solver.sim_time + bc_time_offset
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

    elif rank == 0:
        logger.info("MPI run finished; final diagnostics will be done in postprocessing.")

finally:
    if hasattr(solver, "start_time_end"):
        solver.log_stats()
