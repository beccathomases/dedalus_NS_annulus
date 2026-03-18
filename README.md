# Navier–Stokes in a 2D Annulus (Dedalus v3)

This repository contains Dedalus v3 scripts for solving the **2D incompressible Navier–Stokes equations** in an **annulus** using polar coordinates `(phi, r)`.

The main production script at present is:

- `NS_annulus_mpi.py` — MPI-capable annulus solver with segmented runs, restart options, checkpointing, and postprocessing-friendly output

Other scripts in the repository include earlier variants, restart/refinement utilities, and analysis/postprocessing scripts.

---

## Geometry and discretization

The problem is posed in polar coordinates on an annulus:

- `phi in [0, 2*pi)` — Fourier direction
- `r in [Ri, Ro]` — radial spectral direction via `AnnulusBasis`

The solver uses Dedalus’ **two-boundary tau formulation**, appropriate for annular domains with inner and outer boundaries.

For `PolarCoordinates('phi','r')`, the velocity component ordering is:

- `u['g'][0]` = azimuthal component `u_phi`
- `u['g'][1]` = radial component `u_r`

---

## Equations

We solve the incompressible Navier–Stokes equations

- `div(u) = 0`
- `dt(u) + u·grad(u) = -grad(p) + nu * div(grad_u)`

using a first-order viscous formulation with tau lifting:

- `grad_u = grad(u) + rvec * Lift(tau_u1)`

In the code, the IVP is written as

- `trace(grad_u) + tau_p = 0`
- `dt(u) - nu*div(grad_u) + grad(p) + lift(tau_u2) = - u@grad(u)`

A pressure gauge is imposed to remove the nullspace:

- `integ(p) = 0`

---

## Boundary conditions

### Inner wall: no slip
At the inner boundary `r = Ri`,

- `u(r=Ri) = 0`

### Outer wall: prescribed oscillatory horizontal translation
At the outer boundary `r = Ro`, the wall undergoes a time-dependent horizontal translation with

- `u_x(t) = Amp * pi * [cos(2*pi*t) + 2*cos(4*pi*t)]`
- `u_y(t) = 0`

This is imposed in polar components as

- `u_r(phi,t)   =  u_x(t) cos(phi)`
- `u_phi(phi,t) = -u_x(t) sin(phi)`

The helper `set_outer_bc(t)` updates the boundary field `u_outer` at each step.

---

## Reynolds number convention

The script uses the convention

- `Rref = 1`
- `Omega_ref = 2*pi`

and sets the viscosity by

- `nu = Rref^2 * Omega_ref / Re_target`

So in the current setup, with `Rref = 1`, this reduces to

- `nu = 2*pi / Re_target`

---

## Run organization

Runs are organized into case folders and segment folders:

- `case_label/segment_label`

For example:

- `Re40_caseB/seg01_from_rest_snap0p1`
- `Re40_caseA/seg02_refine_snap0p01`

Each segment writes to its own output directory:

- `run_dir/snapshots/`
- `run_dir/checkpoints/`
- `run_dir/run_info.txt`

This makes it easier to run long jobs in pieces, continue from checkpoints, and refine or restart from saved states.

---

## Restart modes

The main script supports several restart options:

- `restart_mode = "none"`
  - start from rest

- `restart_mode = "full"`
  - exact continuation from a checkpoint file using `solver.load_state(...)`

- `restart_mode = "u_only"`
  - load velocity only from saved output, then reset pressure and tau fields

- `restart_mode = "u_refine"`
  - load saved velocity from a previous run and interpolate it onto a new grid using `RegularGridInterpolator`

For restart-based runs, the script also tracks a boundary-condition time offset so that the forcing stays synchronized with the physical time of the saved data.

---

## Startup ramp

For fresh runs from rest, the outer-wall forcing can be ramped on smoothly using

- `ramp(t) = 1 - exp(-t / ramp_time)`

By default:

- the ramp is used when `restart_mode == "none"`
- the ramp is disabled automatically for restart runs

This helps avoid an abrupt startup transient when beginning from rest.

---

## Outputs

### Snapshots
The script writes snapshot tasks to

- `run_dir/snapshots/`

including:

- `u`
- `p`
- `vorticity`
- `divu`

These are written every `snapshot_dt`.

### Checkpoints
The script writes checkpoint files to

- `run_dir/checkpoints/`

These contain the full solver state and are intended for restart/continuation.

### Run metadata
Each segment also writes

- `run_dir/run_info.txt`

which records key parameters such as:

- case/segment labels
- Reynolds number
- grid resolution
- geometry
- viscosity
- forcing amplitude
- restart settings
- output cadence
- physical start/end times

---

## Runtime guardrails

The script includes several lightweight runtime checks:

- an early boundary-condition sanity check
- periodic non-finite-value checks on key fields
- minimal progress logging (`iteration`, `time`, `dt`)

If non-finite values are detected, the run stops early and skips final summary diagnostics.

---

## End-of-run behavior

For serial runs (`size == 1`), the script computes some final diagnostics, including:

- Fourier mode analysis of the Cartesian `u_x` field
- saved comparison data in `compare_m1_ux_final.npz`
- RMS divergence
- RMS vorticity
- kinetic energy
- enstrophy
- boundary-condition mismatch at the outer wall
- inner-wall slip
- net outer flux

For MPI runs, the code currently leaves detailed diagnostics to postprocessing and logs only a summary message on rank 0.

---

## Main parameters to edit

The most important user-editable parameters near the top of `NS_annulus_mpi.py` are:

- `Re_target` — target Reynolds number
- `case_label`, `segment_label` — output folder organization
- `restart_mode`, `restart_file`, `restart_index` — restart behavior
- `segment_duration` — how much additional simulation time to run
- `snapshot_dt`, `checkpoint_dt` — output cadence
- `max_dt` — fixed timestep
- `Ri`, `Ro` — inner/outer radii
- `Nphi`, `Nr` — grid resolution
- `Amp` — forcing amplitude
- `ramp_time` — startup ramp timescale

---

## Running

### Serial
```bash
python NS_annulus_mpi.py
```

### MPI
```bash
mpiexec -n 4 python NS_annulus_mpi.py
```

On a cluster using Slurm, this would typically be launched with srun inside a job script.

### Repository notes

This repository currently contains a mixture of:

 - main solver scripts

 - alternate solver variants

- restart/refinement variants

- analysis and comparison scripts

- MATLAB helper scripts for visualization/postprocessing

Large simulation outputs, checkpoint folders, movies, figures, and local data files are intended to be ignored via .gitignore.
