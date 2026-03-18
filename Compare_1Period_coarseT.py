import numpy as np
import h5py
import glob
import os
from collections import deque
import matplotlib.pyplot as plt
import csv

# ============================================================
# User settings
# ============================================================
snap_dir = "Re40_caseA/seg01_from_rest_snap0p1/snapshots"   # folder containing snapshots_s*.h5
forcing_period = 1.0
snapshot_dt = 0.1                     # what you used in the run
lag_steps = int(round(forcing_period / snapshot_dt))

csv_out = os.path.join(snap_dir, "period_convergence.csv")
fig_out = os.path.join(snap_dir, "period_convergence.png")

# ============================================================
# Helpers
# ============================================================
def get_u_components(uarr):
    """
    Accept either shape (2,nphi,nr) or (nphi,nr,2).
    Return uphi, ur.
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

def rel_change(new, old):
    return (new - old) / max(abs(old), 1e-300)

def snapshot_file_key(path):
    # sorts snapshots_s1.h5, snapshots_s2.h5, ...
    base = os.path.basename(path)
    num = ''.join(ch for ch in base if ch.isdigit())
    return int(num) if num else 0

# ============================================================
# Find snapshot files
# ============================================================
files = sorted(glob.glob(os.path.join(snap_dir, "snapshots_s*.h5")), key=snapshot_file_key)
if not files:
    raise FileNotFoundError(f"No snapshot files found in {snap_dir}")

print("Found files:")
for f in files:
    print("  ", f)
print(f"Using lag_steps = {lag_steps} for period = {forcing_period} and snapshot_dt = {snapshot_dt}")

# ============================================================
# Stream through snapshots and compare one period apart
# ============================================================
history = deque()
rows = []

for fname in files:
    with h5py.File(fname, "r") as f:
        times = np.array(f["scales/sim_time"])
        u_task = f["tasks/u"]
        p_task = f["tasks/p"]
        om_task = f["tasks/vorticity"]

        for j, t in enumerate(times):
            u_snap = np.array(u_task[j])
            p_snap = np.array(p_task[j]).squeeze()
            om_snap = np.array(om_task[j]).squeeze()

            uphi, ur = get_u_components(u_snap)

            speed = np.sqrt(uphi**2 + ur**2)
            max_u = float(np.max(speed))
            ke_proxy = 0.5 * float(np.mean(speed**2))
            ens_proxy = 0.5 * float(np.mean(om_snap**2))

            cur = {
                "t": float(t),
                "uphi": uphi.copy(),
                "ur": ur.copy(),
                "p": p_snap.copy(),
                "omega": om_snap.copy(),
                "max_u": max_u,
                "KE_proxy": ke_proxy,
                "Ens_proxy": ens_proxy,
            }

            if len(history) >= lag_steps:
                old = history[0]

                row = {
                    "t": cur["t"],
                    "t_old": old["t"],
                    "dt_period": cur["t"] - old["t"],
                    "relL2_u": rel_l2_vec(cur["uphi"], cur["ur"], old["uphi"], old["ur"]),
                    "relL2_p": rel_l2(cur["p"], old["p"]),
                    "relL2_omega": rel_l2(cur["omega"], old["omega"]),
                    "dmax_u": rel_change(cur["max_u"], old["max_u"]),
                    "dKE_proxy": rel_change(cur["KE_proxy"], old["KE_proxy"]),
                    "dEns_proxy": rel_change(cur["Ens_proxy"], old["Ens_proxy"]),
                    "max_u": cur["max_u"],
                    "KE_proxy": cur["KE_proxy"],
                    "Ens_proxy": cur["Ens_proxy"],
                }
                rows.append(row)

            history.append(cur)
            if len(history) > lag_steps:
                history.popleft()

# ============================================================
# Save CSV
# ============================================================
if not rows:
    raise RuntimeError("Not enough snapshots to compare one period apart.")

fieldnames = list(rows[0].keys())
with open(csv_out, "w", newline="") as fp:
    writer = csv.DictWriter(fp, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f"\nSaved table to: {csv_out}")

# ============================================================
# Print tail summary
# ============================================================
print("\nLast few period-to-period comparisons:")
for row in rows[-10:]:
    print(
        f"t={row['t']:.3f} vs {row['t_old']:.3f}  "
        f"relL2_u={row['relL2_u']:.3e}  "
        f"relL2_p={row['relL2_p']:.3e}  "
        f"relL2_omega={row['relL2_omega']:.3e}  "
        f"dKE={row['dKE_proxy']:+.3e}  "
        f"dEns={row['dEns_proxy']:+.3e}"
    )

# ============================================================
# Plot
# ============================================================
rows_plot = [r for r in rows if r["t_old"] >= 1.0]
# or, simplest version:
# rows_plot = rows[1:]

tvals = np.array([r["t"] for r in rows_plot])
rel_u = np.array([r["relL2_u"] for r in rows_plot])
rel_p = np.array([r["relL2_p"] for r in rows_plot])
rel_om = np.array([r["relL2_omega"] for r in rows_plot])
dke = np.array([abs(r["dKE_proxy"]) for r in rows_plot])
dens = np.array([abs(r["dEns_proxy"]) for r in rows_plot])

fig, axs = plt.subplots(2, 2, figsize=(10, 8))

axs[0, 0].semilogy(tvals, rel_u, marker='o')
axs[0, 0].set_title("relL2_u(t, t-1)")
axs[0, 0].set_xlabel("t")
axs[0, 0].grid(True)

axs[0, 1].semilogy(tvals, rel_om, marker='o')
axs[0, 1].set_title("relL2_omega(t, t-1)")
axs[0, 1].set_xlabel("t")
axs[0, 1].grid(True)

axs[1, 0].semilogy(tvals, rel_p, marker='o')
axs[1, 0].set_title("relL2_p(t, t-1)")
axs[1, 0].set_xlabel("t")
axs[1, 0].grid(True)

axs[1, 1].semilogy(tvals, dke, marker='o', label='|dKE_proxy|')
axs[1, 1].semilogy(tvals, dens, marker='o', label='|dEns_proxy|')
axs[1, 1].set_title("period-to-period scalar changes")
axs[1, 1].set_xlabel("t")
axs[1, 1].grid(True)
axs[1, 1].legend()

plt.tight_layout()
plt.savefig(fig_out, dpi=150)
plt.show()

print(f"Saved figure to: {fig_out}")