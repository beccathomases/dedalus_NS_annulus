# Navier–Stokes in a 2D Annulus (Dedalus v3)

This repository contains a Dedalus v3 (`dedalus.public as d3`) script that solves the **2D incompressible Navier–Stokes equations** in an **annulus** using polar coordinates `(phi, r)`.

## Domain / discretization

- Azimuth: `phi ∈ [0, 2π)` (Fourier basis)
- Radius: `r ∈ [Ri, Ro]` (radial spectral basis via `AnnulusBasis`)

The implementation uses Dedalus’ **two-boundary tau formulation** appropriate for annular domains.

---

## Files

- `NS_annulus.py` — main solver script
- `snapshots/` — Dedalus output directory (HDF5), containing:
  - velocity `u`
  - pressure `p`
  - vorticity `vorticity`
  - divergence `divu`

---

## Model

We solve (vector form)

- Incompressibility: `div(u) = 0`
- Momentum:
  \[
  \partial_t u + u\cdot\nabla u = -\nabla p + \nu \nabla^2 u
  \]
with kinematic viscosity `nu = 1/Re`.

In the script, the IVP is written as:
- `trace(grad_u) + tau_p = 0`
- `dt(u) - nu*div(grad_u) + grad(p) + lift(tau_u2) = - u@grad(u)`

The viscous term is handled in first-order form using tau lifting:
- `grad_u = grad(u) + rvec * lift(tau_u1)`

A pressure gauge is applied to remove the nullspace:
- `integ(p) = 0`

---

## Boundary conditions

- Inner wall (`r = Ri`): **no slip**
  - `u(r=Ri) = 0`

- Outer wall (`r = Ro`): **prescribed velocity**
  - `u(r=Ro) = u_outer`

### Current outer-wall driving: Couette rotation

The helper `set_outer_bc(t)` currently prescribes rigid rotation of the outer cylinder:
- angular speed: `Omega_out`
- outer-wall tangential speed: `Uo = Omega_out * Ro`

Implemented as:
- `u_outer['g'][0] = Uo` (azimuthal component `u_phi`)
- `u_outer['g'][1] = 0`  (radial component `u_r`)

**Component ordering:** For `PolarCoordinates('phi','r')`,
- `u['g'][0]` is azimuthal (`u_phi`)
- `u['g'][1]` is radial (`u_r`)

---

## Initial condition

The script currently initializes with the **exact laminar Couette profile** (phi-independent),
which is a steady solution for the Couette boundary forcing:

\[
u_\phi(r) = \Omega_{out}\,\frac{R_o^2}{R_o^2-R_i^2}\left(r-\frac{R_i^2}{r}\right),
\qquad u_r = 0
\]

This IC makes the script a quick correctness/regression check for the annulus/tau setup.

To run a **spin-up transient**, replace the IC block with rest (or small noise).

---

## Diagnostics

Printed every ~50 iterations:
- `max|u|`
- `rms(div)` computed from \(\sqrt{\int (div\,u)^2 / \text{area}}\)
- `rms(ω)` computed from \(\sqrt{\int ω^2 / \text{area}}\)

Vorticity is computed as a scalar (out-of-plane) quantity:
- `omega = -div(skew(u))`

At the end of the run the script computes the **relative L2 error** of the phi-averaged
azimuthal velocity profile vs. the analytic Couette solution.

---

## Running

Activate your Dedalus environment and run:

```bash
python NS_annulus.py

```

## Parameters (edit in the script)

Key parameters near the top of NS_annulus.py:

- Ri, Ro — inner/outer radii

- Nphi, Nr — spatial resolution

- Re and nu = 1/Re — Reynolds number / kinematic viscosity

- stop_sim_time — simulation end time

- max_dt and CFL settings — timestep control

- Omega_out — outer cylinder rotation rate (used in set_outer_bc)

## Notes / assumptions

This is a 2D polar-annulus model (no axial dependence).

The tau formulation uses outer_edge as the 1D boundary basis container for tau fields and for the boundary data field u_outer.

