import numpy as np
import h5py
import glob
import os
import csv
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator

# ============================================================
# User settings
# ============================================================
snap_dir = "Re40_10per_dt01/snapshots"   # folder containing snapshots_s*.h5
forcing_period = 1.0

# Geometry / parameters used in the run
Ri = 1.0
Ro = 8.0
Re_target = 40.0
Omega_ref = 2*np.pi
Rref = 1.0
nu = (Rref**2) * Omega_ref / Re_target

# Which window to analyze:
# By default: use the last full forcing period in the data
use_last_period = True
t_start_user = None
t_end_user = None

# Cartesian grid for period-averaged streamline plot
cart_N = 300

# Outputs
force_csv = os.path.join(snap_dir, "refined_force_timeseries.csv")
summary_txt = os.path.join(snap_dir, "refined_period_summary.txt")
force_fig = os.path.join(snap_dir, "refined_force_plot.png")
stream_fig = os.path.join(snap_dir, "period_averaged_streamlines.png")

# ============================================================
# Helpers
# ============================================================
def snapshot_file_key(path):
    base = os.path.basename(path)
    num = ''.join(ch for ch in base if ch.isdigit())
    return int(num) if num else 0

def get_u_components(uarr):
    """
    Accept either shape (2,nphi,nr) or (nphi,nr,2).
    Return uphi, ur with shape (nphi, nr).
    """
    if uarr.ndim != 3:
        raise ValueError(f"Expected 3D u snapshot, got shape {uarr.shape}")
    if uarr.shape[0] == 2:
        return uarr[0], uarr[1]
    elif uarr.shape[-1] == 2:
        return uarr[..., 0], uarr[..., 1]
    else:
        raise ValueError(f"Cannot identify component axis in u with shape {uarr.shape}")

def rel_l2(a, b):
    num = np.sum((a - b)**2, dtype=np.float64)
    den = np.sum(b**2, dtype=np.float64)
    return np.sqrt(num / max(den, 1e-300))

def rel_l2_vec(a1, a2, b1, b2):
    num = np.sum((a1 - b1)**2 + (a2 - b2)**2, dtype=np.float64)
    den = np.sum(b1**2 + b2**2, dtype=np.float64)
    return np.sqrt(num / max(den, 1e-300))

def walk_scale_datasets(group):
    out = []
    for key, val in group.items():
        if isinstance(val, h5py.Dataset):
            arr = np.array(val).squeeze()
            if arr.ndim == 1 and arr.size > 1:
                out.append((val.name, arr))
        elif isinstance(val, h5py.Group):
            out.extend(walk_scale_datasets(val))
    return out

def guess_phi_r_from_file(f, nphi, nr, Ri, Ro):
    """
    Try to recover phi and r coordinate arrays from the HDF5 scales.
    Falls back to uniform phi if needed.
    """
    if "scales" not in f:
        raise RuntimeError("No /scales group found in file.")

    cands = walk_scale_datasets(f["scales"])

    phi = None
    r = None

    # First pass: by name
    for name, arr in cands:
        lname = name.lower()
        if "phi" in lname and arr.size == nphi:
            phi = arr.copy()
        if lname.endswith("/r") and arr.size == nr:
            r = arr.copy()

    # Second pass: by value range / size
    if phi is None:
        phi_opts = []
        for name, arr in cands:
            if arr.size == nphi:
                span = np.nanmax(arr) - np.nanmin(arr)
                if span > 5.0:
                    phi_opts.append(arr)
        if phi_opts:
            phi = phi_opts[0].copy()

    if r is None:
        r_opts = []
        for name, arr in cands:
            if arr.size == nr:
                amin = np.nanmin(arr)
                amax = np.nanmax(arr)
                if amin >= 0 and amax > amin:
                    r_opts.append(arr)
        if r_opts:
            # choose candidate whose span best matches Ro-Ri
            spans = [abs((arr.max() - arr.min()) - (Ro - Ri)) for arr in r_opts]
            r = r_opts[int(np.argmin(spans))].copy()

    # Fallback phi
    if phi is None:
        phi = np.linspace(0, 2*np.pi, nphi, endpoint=False)

    if r is None:
        raise RuntimeError("Could not determine radial coordinate array from HDF5 file.")

    phi = np.array(phi).squeeze()
    r = np.array(r).squeeze()

    # Sort if needed
    if np.any(np.diff(phi) < 0):
        idx = np.argsort(phi)
        phi = phi[idx]
    if np.any(np.diff(r) < 0):
        idx = np.argsort(r)
        r = r[idx]

    return phi, r

def periodic_phi_derivative(q, phi):
    """
    Periodic centered derivative in phi for q(phi,r).
    q shape: (nphi, nr)
    """
    dphi = phi[1] - phi[0]
    return (np.roll(q, -1, axis=0) - np.roll(q, 1, axis=0)) / (2*dphi)

def radial_derivative(q, r):
    """
    Radial derivative for q(phi,r), nonuniform r allowed.
    """
    return np.gradient(q, r, axis=1, edge_order=2)

def cylinder_force_from_snapshot(uphi, ur, p, phi, r, Ri, nu):
    """
    Compute net force on the inner cylinder from one snapshot.

    Uses traction on the cylinder with normal pointing from cylinder into fluid (+e_r).
    If you want the opposite sign convention, flip Fx and Fy.
    """
    dphi_ur = periodic_phi_derivative(ur, phi)
    dr_ur = radial_derivative(ur, r)
    dr_uphi = radial_derivative(uphi, r)

    # inner boundary index
    j0 = 0
    rr = r[j0]

    # stress components at r = Ri
    sigma_rr = -p[:, j0] + 2.0 * nu * dr_ur[:, j0]
    sigma_phir = nu * ((1.0/rr) * dphi_ur[:, j0] + dr_uphi[:, j0] - uphi[:, j0]/rr)

    fx_density = sigma_rr * np.cos(phi) - sigma_phir * np.sin(phi)
    fy_density = sigma_rr * np.sin(phi) + sigma_phir * np.cos(phi)

    dphi = phi[1] - phi[0]
    Fx = rr * dphi * np.sum(fx_density)
    Fy = rr * dphi * np.sum(fy_density)

    return Fx, Fy

def periodic_extension(phi, q):
    """
    Extend q(phi,...) periodically by one row in phi.
    q shape: (nphi, nr)
    """
    phi_ext = np.concatenate([phi, [phi[0] + 2*np.pi]])
    q_ext = np.concatenate([q, q[0:1, :]], axis=0)
    return phi_ext, q_ext

# ============================================================
# Find files and collect snapshot times
# ============================================================
files = sorted(glob.glob(os.path.join(snap_dir, "snapshots_s*.h5")), key=snapshot_file_key)
if not files:
    raise FileNotFoundError(f"No snapshot files found in {snap_dir}")

records = []   # (time, filename, local_index)
for fname in files:
    with h5py.File(fname, "r") as f:
        times = np.array(f["scales/sim_time"])
        for j, t in enumerate(times):
            records.append((float(t), fname, j))

records.sort(key=lambda x: x[0])
all_times = np.array([r[0] for r in records])

if use_last_period:
    t_end = all_times[-1]
    t_start = t_end - forcing_period
else:
    if t_start_user is None or t_end_user is None:
        raise ValueError("Set t_start_user and t_end_user, or use use_last_period=True.")
    t_start = t_start_user
    t_end = t_end_user

tol = 1e-12
selected = [rec for rec in records if (t_start - tol) <= rec[0] <= (t_end + tol)]

if len(selected) < 2:
    raise RuntimeError("Not enough snapshots in selected analysis window.")

print(f"Using analysis window [{t_start:.6f}, {t_end:.6f}]")
print(f"Number of selected snapshots: {len(selected)}")

# ============================================================
# Get coordinates from first selected snapshot file
# ============================================================
with h5py.File(selected[0][1], "r") as f:
    u0 = np.array(f["tasks/u"][selected[0][2]])
    uphi0, ur0 = get_u_components(u0)
    nphi, nr = uphi0.shape
    phi, r = guess_phi_r_from_file(f, nphi, nr, Ri, Ro)

# ============================================================
# Loop through selected snapshots:
#   - force time series
#   - trapezoidal time average of u and p and omega
#   - start/end period mismatch
# ============================================================
force_rows = []

first = None
prev = None

int_uphi = None
int_ur = None
int_p = None
int_omega = None
int_Fx = 0.0
int_Fy = 0.0

for t, fname, j in selected:
    with h5py.File(fname, "r") as f:
        u_snap = np.array(f["tasks/u"][j])
        p_snap = np.array(f["tasks/p"][j]).squeeze()
        om_snap = np.array(f["tasks/vorticity"][j]).squeeze()

    uphi, ur = get_u_components(u_snap)

    Fx, Fy = cylinder_force_from_snapshot(uphi, ur, p_snap, phi, r, Ri, nu)

    speed = np.sqrt(uphi**2 + ur**2)
    force_rows.append({
        "t": float(t),
        "Fx": float(Fx),
        "Fy": float(Fy),
        "Fmag": float(np.sqrt(Fx**2 + Fy**2)),
        "max_u": float(np.max(speed)),
    })

    cur = {
        "t": float(t),
        "uphi": uphi,
        "ur": ur,
        "p": p_snap,
        "omega": om_snap,
        "Fx": Fx,
        "Fy": Fy,
    }

    if first is None:
        first = cur

    if prev is not None:
        dt = cur["t"] - prev["t"]

        if int_uphi is None:
            int_uphi = 0.5 * dt * (prev["uphi"] + cur["uphi"])
            int_ur   = 0.5 * dt * (prev["ur"]   + cur["ur"])
            int_p    = 0.5 * dt * (prev["p"]    + cur["p"])
            int_omega= 0.5 * dt * (prev["omega"]+ cur["omega"])
        else:
            int_uphi += 0.5 * dt * (prev["uphi"] + cur["uphi"])
            int_ur   += 0.5 * dt * (prev["ur"]   + cur["ur"])
            int_p    += 0.5 * dt * (prev["p"]    + cur["p"])
            int_omega+= 0.5 * dt * (prev["omega"]+ cur["omega"])

        int_Fx += 0.5 * dt * (prev["Fx"] + cur["Fx"])
        int_Fy += 0.5 * dt * (prev["Fy"] + cur["Fy"])

    prev = cur

last = prev
T_used = last["t"] - first["t"]
if T_used <= 0:
    raise RuntimeError("Selected analysis window has zero or negative duration.")

avg_uphi = int_uphi / T_used
avg_ur   = int_ur   / T_used
avg_p    = int_p    / T_used
avg_omega= int_omega/ T_used
avg_Fx   = int_Fx   / T_used
avg_Fy   = int_Fy   / T_used
avg_Fmag = np.sqrt(avg_Fx**2 + avg_Fy**2)

# ============================================================
# Start/end mismatch over one period
# ============================================================
rel_u = rel_l2_vec(last["uphi"], last["ur"], first["uphi"], first["ur"])
rel_p = rel_l2(last["p"], first["p"])
rel_omega = rel_l2(last["omega"], first["omega"])

# ============================================================
# Save force time series CSV
# ============================================================
with open(force_csv, "w", newline="") as fp:
    writer = csv.DictWriter(fp, fieldnames=list(force_rows[0].keys()))
    writer.writeheader()
    writer.writerows(force_rows)

# ============================================================
# Write text summary
# ============================================================
with open(summary_txt, "w") as fp:
    fp.write(f"Analysis window: [{first['t']:.12f}, {last['t']:.12f}]\n")
    fp.write(f"Window length used: {T_used:.12f}\n\n")

    fp.write("Start/end mismatch over selected period:\n")
    fp.write(f"  relL2_u      = {rel_u:.6e}\n")
    fp.write(f"  relL2_p      = {rel_p:.6e}\n")
    fp.write(f"  relL2_omega  = {rel_omega:.6e}\n\n")

    fp.write("Period-averaged force on inner cylinder:\n")
    fp.write(f"  <Fx>         = {avg_Fx:.6e}\n")
    fp.write(f"  <Fy>         = {avg_Fy:.6e}\n")
    fp.write(f"  |<F>|        = {avg_Fmag:.6e}\n")

print("\nSummary:")
print(f"Window used: t = {first['t']:.6f} to {last['t']:.6f}")
print(f"relL2_u(start,end)      = {rel_u:.6e}")
print(f"relL2_p(start,end)      = {rel_p:.6e}")
print(f"relL2_omega(start,end)  = {rel_omega:.6e}")
print(f"<Fx> = {avg_Fx:.6e}, <Fy> = {avg_Fy:.6e}, |<F>| = {avg_Fmag:.6e}")
print(f"Saved force CSV to: {force_csv}")
print(f"Saved text summary to: {summary_txt}")

# ============================================================
# Plot force time series over selected period
# ============================================================
tvals = np.array([row["t"] for row in force_rows])
Fxvals = np.array([row["Fx"] for row in force_rows])
Fyvals = np.array([row["Fy"] for row in force_rows])
Fmagvals = np.array([row["Fmag"] for row in force_rows])

fig, axs = plt.subplots(2, 1, figsize=(8, 7), sharex=True)

axs[0].plot(tvals, Fxvals, marker='o', label='Fx')
axs[0].plot(tvals, Fyvals, marker='o', label='Fy')
axs[0].axhline(avg_Fx, linestyle='--', linewidth=1, label='<Fx>')
axs[0].axhline(avg_Fy, linestyle='--', linewidth=1, label='<Fy>')
axs[0].set_ylabel("force")
axs[0].set_title("Net force on inner cylinder over selected period")
axs[0].grid(True)
axs[0].legend()

axs[1].plot(tvals, Fmagvals, marker='o', label='|F|')
axs[1].axhline(avg_Fmag, linestyle='--', linewidth=1, label='|<F>|')
axs[1].set_xlabel("t")
axs[1].set_ylabel("|force|")
axs[1].grid(True)
axs[1].legend()

plt.tight_layout()
plt.savefig(force_fig, dpi=150)
plt.show()

print(f"Saved force figure to: {force_fig}")

# ============================================================
# Period-averaged flow: Cartesian streamline plot
# ============================================================
phi2, r2 = np.meshgrid(phi, r, indexing='ij')

ux_avg = avg_ur * np.cos(phi2) - avg_uphi * np.sin(phi2)
uy_avg = avg_ur * np.sin(phi2) + avg_uphi * np.cos(phi2)

phi_ext, ux_ext = periodic_extension(phi, ux_avg)
_, uy_ext = periodic_extension(phi, uy_avg)

interp_ux = RegularGridInterpolator(
    (phi_ext, r), ux_ext, method="linear", bounds_error=False, fill_value=np.nan
)
interp_uy = RegularGridInterpolator(
    (phi_ext, r), uy_ext, method="linear", bounds_error=False, fill_value=np.nan
)

xg = np.linspace(-Ro, Ro, cart_N)
yg = np.linspace(-Ro, Ro, cart_N)
X, Y = np.meshgrid(xg, yg)

R = np.sqrt(X**2 + Y**2)
PHI = np.mod(np.arctan2(Y, X), 2*np.pi)

pts = np.column_stack([PHI.ravel(), R.ravel()])
Ux = interp_ux(pts).reshape(X.shape)
Uy = interp_uy(pts).reshape(X.shape)

mask = (R < Ri) | (R > Ro)
Ux = np.ma.array(Ux, mask=mask)
Uy = np.ma.array(Uy, mask=mask)
Speed = np.ma.array(np.sqrt(Ux**2 + Uy**2), mask=mask)

fig2, ax = plt.subplots(figsize=(8, 8))
cf = ax.contourf(X, Y, Speed, levels=50)
plt.colorbar(cf, ax=ax, label="|<u>| over one period")

ax.streamplot(xg, yg, Ux, Uy, density=1.8, linewidth=1.0, arrowsize=1.0)

inner = plt.Circle((0, 0), Ri, color='white', zorder=10)
outer = plt.Circle((0, 0), Ro, fill=False, color='k', linewidth=1.0)
ax.add_patch(inner)
ax.add_patch(outer)

ax.set_aspect('equal')
ax.set_xlim(-Ro, Ro)
ax.set_ylim(-Ro, Ro)
ax.set_title("Streamlines of period-averaged flow")
ax.set_xlabel("x")
ax.set_ylabel("y")

plt.tight_layout()
plt.savefig(stream_fig, dpi=150)
plt.show()

print(f"Saved streamline figure to: {stream_fig}")