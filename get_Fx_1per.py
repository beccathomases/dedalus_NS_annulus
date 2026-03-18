import numpy as np
import h5py
import glob
import os
import csv
import matplotlib.pyplot as plt

# ============================================================
# User settings
# ============================================================
snap_dir = "Re40_caseA/seg02_refine_snap0p01/snapshots"
forcing_period = 1.0
snapshot_dt = 0.01   # saved time spacing used in the run
steps_per_period = int(round(forcing_period / snapshot_dt))

Ri = 1.0
Ro = 8.0
Re_target = 40.0
Omega_ref = 2*np.pi
Rref = 1.0
nu = (Rref**2) * Omega_ref / Re_target

csv_out = os.path.join(snap_dir, "force_period_avg_timeseries.csv")
fig_out = os.path.join(snap_dir, "force_period_avg_timeseries.png")

# ============================================================
# Helpers
# ============================================================
def snapshot_file_key(path):
    base = os.path.basename(path)
    num = ''.join(ch for ch in base if ch.isdigit())
    return int(num) if num else 0

def get_u_components(uarr):
    if uarr.ndim != 3:
        raise ValueError(f"Expected 3D u snapshot, got shape {uarr.shape}")
    if uarr.shape[0] == 2:
        return uarr[0], uarr[1]
    elif uarr.shape[-1] == 2:
        return uarr[..., 0], uarr[..., 1]
    else:
        raise ValueError(f"Cannot identify component axis in u with shape {uarr.shape}")

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
    cands = walk_scale_datasets(f["scales"])

    phi = None
    r = None

    for name, arr in cands:
        lname = name.lower()
        if "phi" in lname and arr.size == nphi:
            phi = arr.copy()
        if lname.endswith("/r") and arr.size == nr:
            r = arr.copy()

    if phi is None:
        phi = np.linspace(0, 2*np.pi, nphi, endpoint=False)

    if r is None:
        r_opts = []
        for name, arr in cands:
            if arr.size == nr:
                amin = np.nanmin(arr)
                amax = np.nanmax(arr)
                if amin >= 0 and amax > amin:
                    r_opts.append(arr)
        if not r_opts:
            raise RuntimeError("Could not determine radial coordinate array from HDF5 file.")
        spans = [abs((arr.max() - arr.min()) - (Ro - Ri)) for arr in r_opts]
        r = r_opts[int(np.argmin(spans))].copy()

    phi = np.array(phi).squeeze()
    r = np.array(r).squeeze()

    if np.any(np.diff(phi) < 0):
        phi = phi[np.argsort(phi)]
    if np.any(np.diff(r) < 0):
        r = r[np.argsort(r)]

    return phi, r

def periodic_phi_derivative(q, phi):
    dphi = phi[1] - phi[0]
    return (np.roll(q, -1, axis=0) - np.roll(q, 1, axis=0)) / (2*dphi)

def radial_derivative(q, r):
    return np.gradient(q, r, axis=1, edge_order=2)

def cylinder_force_from_snapshot(uphi, ur, p, phi, r, Ri, nu):
    dphi_ur = periodic_phi_derivative(ur, phi)
    dr_ur = radial_derivative(ur, r)
    dr_uphi = radial_derivative(uphi, r)

    j0 = 0
    rr = r[j0]

    sigma_rr = -p[:, j0] + 2.0 * nu * dr_ur[:, j0]
    sigma_phir = nu * ((1.0/rr) * dphi_ur[:, j0] + dr_uphi[:, j0] - uphi[:, j0]/rr)

    fx_density = sigma_rr * np.cos(phi) - sigma_phir * np.sin(phi)
    fy_density = sigma_rr * np.sin(phi) + sigma_phir * np.cos(phi)

    dphi = phi[1] - phi[0]
    Fx = rr * dphi * np.sum(fx_density)
    Fy = rr * dphi * np.sum(fy_density)
    return Fx, Fy

def load_records(snap_dir):
    files = sorted(glob.glob(os.path.join(snap_dir, "snapshots_s*.h5")), key=snapshot_file_key)
    if not files:
        raise FileNotFoundError(f"No snapshot files found in {snap_dir}")

    records = []
    for fname in files:
        with h5py.File(fname, "r") as f:
            times = np.array(f["scales/sim_time"])
            for j, t in enumerate(times):
                records.append((float(t), fname, j))

    records.sort(key=lambda x: x[0])

    # remove duplicate times from restart joins
    deduped = []
    last_t = None
    for rec in records:
        t = rec[0]
        if last_t is None or abs(t - last_t) > 1e-12:
            deduped.append(rec)
            last_t = t

    return deduped

def read_snapshot(fname, j):
    with h5py.File(fname, "r") as f:
        u_snap = np.array(f["tasks/u"][j])
        p_snap = np.array(f["tasks/p"][j]).squeeze()
    uphi, ur = get_u_components(u_snap)
    return uphi, ur, p_snap

# ============================================================
# Load records and coordinates
# ============================================================
records = load_records(snap_dir)
all_times = np.array([rec[0] for rec in records])

print("Files being used:")
for f in sorted(set(rec[1] for rec in records), key=snapshot_file_key):
    print(" ", f)

print(f"\nNumber of snapshots: {len(records)}")
print(f"First saved time: {all_times[0]:.6f}")
print(f"Last saved time:  {all_times[-1]:.6f}")
print(f"steps_per_period = {steps_per_period}")

with h5py.File(records[0][1], "r") as f:
    u0 = np.array(f["tasks/u"][records[0][2]])
    uphi0, ur0 = get_u_components(u0)
    nphi, nr = uphi0.shape
    phi, r = guess_phi_r_from_file(f, nphi, nr, Ri, Ro)

# ============================================================
# Instantaneous force time series
# ============================================================
times = []
Fx_vals = []
Fy_vals = []

for t, fname, j in records:
    uphi, ur, p_snap = read_snapshot(fname, j)
    Fx, Fy = cylinder_force_from_snapshot(uphi, ur, p_snap, phi, r, Ri, nu)
    times.append(t)
    Fx_vals.append(Fx)
    Fy_vals.append(Fy)

times = np.array(times)
Fx_vals = np.array(Fx_vals)
Fy_vals = np.array(Fy_vals)

# ============================================================
# Running one-period average using trapezoidal rule
# Fxbar(t_k) = (1/T) \int_{t_k-T}^{t_k} Fx(s) ds
# ============================================================
Fxbar_times = []
Fxbar_vals = []
Fybar_vals = []

for k in range(steps_per_period, len(times)):
    t_window = times[k-steps_per_period:k+1]
    Fx_window = Fx_vals[k-steps_per_period:k+1]
    Fy_window = Fy_vals[k-steps_per_period:k+1]

    T_used = t_window[-1] - t_window[0]
    if T_used <= 0:
        continue

    Fxbar = np.trapezoid(Fx_window, t_window) / T_used
    Fybar = np.trapezoid(Fy_window, t_window) / T_used

    Fxbar_times.append(times[k])
    Fxbar_vals.append(Fxbar)
    Fybar_vals.append(Fybar)

Fxbar_times = np.array(Fxbar_times)
Fxbar_vals = np.array(Fxbar_vals)
Fybar_vals = np.array(Fybar_vals)

# ============================================================
# Save CSV
# ============================================================
with open(csv_out, "w", newline="") as fp:
    writer = csv.DictWriter(
        fp,
        fieldnames=["t", "Fx", "Fy", "Fxbar_1T", "Fybar_1T"]
    )
    writer.writeheader()

    fxbar_map = {float(t): (float(fx), float(fy)) for t, fx, fy in zip(Fxbar_times, Fxbar_vals, Fybar_vals)}

    for t, fx, fy in zip(times, Fx_vals, Fy_vals):
        row = {
            "t": float(t),
            "Fx": float(fx),
            "Fy": float(fy),
            "Fxbar_1T": "",
            "Fybar_1T": "",
        }
        if float(t) in fxbar_map:
            row["Fxbar_1T"] = fxbar_map[float(t)][0]
            row["Fybar_1T"] = fxbar_map[float(t)][1]
        writer.writerow(row)

print(f"\nSaved CSV to: {csv_out}")

# ============================================================
# Plot
# ============================================================
fig, axs = plt.subplots(2, 1, figsize=(9, 8), sharex=True)

axs[0].plot(times, Fx_vals, label="Fx(t)")
axs[0].plot(Fxbar_times, Fxbar_vals, linewidth=2, label="period-avg Fx")
axs[0].set_ylabel("horizontal force")
axs[0].set_title("Instantaneous and running 1-period averaged horizontal force")
axs[0].grid(True)
axs[0].legend()

axs[1].plot(times, Fy_vals, label="Fy(t)")
axs[1].plot(Fxbar_times, Fybar_vals, linewidth=2, label="period-avg Fy")
axs[1].set_xlabel("t")
axs[1].set_ylabel("vertical force")
axs[1].grid(True)
axs[1].legend()

plt.tight_layout()
plt.savefig(fig_out, dpi=150)
plt.show()

print(f"Saved figure to: {fig_out}")

if len(Fxbar_vals) > 0:
    print("\nLast few period-averaged horizontal forces:")
    for t, fxb in zip(Fxbar_times[-10:], Fxbar_vals[-10:]):
        print(f"t={t:.3f}   Fxbar_1T={fxb:+.6e}")