import numpy as np
import h5py
import glob
import os

# ============================================================
# User settings
# ============================================================
snap_dir = "Re40_10per_dt01/snapshots"
T = 1.0
dt_save = 0.01
steps_per_period = int(round(T / dt_save))
avg_periods = 2   # or 3

# ============================================================
# Helpers
# ============================================================
def snapshot_file_key(path):
    base = os.path.basename(path)
    digits = ''.join(ch for ch in base if ch.isdigit())
    return int(digits) if digits else 0

def get_u_components(uarr):
    if uarr.shape[0] == 2:
        return uarr[0], uarr[1]
    elif uarr.shape[-1] == 2:
        return uarr[..., 0], uarr[..., 1]
    else:
        raise ValueError(f"Unexpected u shape {uarr.shape}")

def rel_l2(a, b):
    num = np.sum((a - b)**2, dtype=np.float64)
    den = np.sum(b**2, dtype=np.float64)
    return np.sqrt(num / max(den, 1e-300))

def rel_l2_vec(a1, a2, b1, b2):
    num = np.sum((a1 - b1)**2 + (a2 - b2)**2, dtype=np.float64)
    den = np.sum(b1**2 + b2**2, dtype=np.float64)
    return np.sqrt(num / max(den, 1e-300))

# ============================================================
# Find files
# ============================================================
files = sorted(glob.glob(os.path.join(snap_dir, "snapshots_s*.h5")), key=snapshot_file_key)
if not files:
    raise FileNotFoundError(f"No snapshot files found in {snap_dir}")

print("Files being used:")
for f in files:
    print(" ", f)

# ============================================================
# Read all snapshots into one timeline
# ============================================================
times_all = []
U_all = []
P_all = []
W_all = []

for fname in files:
    with h5py.File(fname, "r") as f:
        times = np.array(f["scales/sim_time"])
        U = f["tasks/u"]
        P = f["tasks/p"]
        W = f["tasks/vorticity"]

        for j in range(len(times)):
            times_all.append(float(times[j]))
            U_all.append(np.array(U[j]))
            P_all.append(np.array(P[j]).squeeze())
            W_all.append(np.array(W[j]).squeeze())

times_all = np.array(times_all)

# Optional: sort by time just to be safe
perm = np.argsort(times_all)
times_all = times_all[perm]
U_all = [U_all[i] for i in perm]
P_all = [P_all[i] for i in perm]
W_all = [W_all[i] for i in perm]

print(f"\nNumber of snapshots: {len(times_all)}")
print(f"First saved time: {times_all[0]:.6f}")
print(f"Last saved time:  {times_all[-1]:.6f}")
print(f"steps_per_period = {steps_per_period}")
print(f"Approx total time covered = {times_all[-1] - times_all[0]:.6f}")

# ============================================================
# Break into period chunks
# ============================================================
n_periods = len(times_all) // steps_per_period
leftover = len(times_all) - n_periods * steps_per_period

print(f"\nn_periods from integer chunking = {n_periods}")
print(f"leftover snapshots = {leftover}")

if n_periods < 2:
    raise RuntimeError("Need at least 2 full periods of saved data.")

period_avgs = []

for k in range(n_periods):
    i0 = k * steps_per_period
    i1 = (k + 1) * steps_per_period

    print(
        f"Period chunk {k}: indices [{i0}:{i1-1}]  "
        f"times [{times_all[i0]:.6f}, {times_all[i1-1]:.6f}]  "
        f"span={times_all[i1-1] - times_all[i0]:.6f}"
    )

    uphi_sum = None
    ur_sum = None
    p_sum = None
    w_sum = None

    for j in range(i0, i1):
        uphi, ur = get_u_components(U_all[j])
        p_snap = P_all[j]
        w_snap = W_all[j]

        if uphi_sum is None:
            uphi_sum = np.zeros_like(uphi)
            ur_sum   = np.zeros_like(ur)
            p_sum    = np.zeros_like(p_snap)
            w_sum    = np.zeros_like(w_snap)

        uphi_sum += uphi
        ur_sum   += ur
        p_sum    += p_snap
        w_sum    += w_snap

    period_avgs.append({
        "k": k,
        "t0": times_all[i0],
        "t1": times_all[i1-1],
        "uphi_bar": uphi_sum / steps_per_period,
        "ur_bar":   ur_sum   / steps_per_period,
        "p_bar":    p_sum    / steps_per_period,
        "w_bar":    w_sum    / steps_per_period,
    })

# ============================================================
# Compare consecutive period averages
# ============================================================
print("\nComparisons:")
for k in range(1, len(period_avgs)):
    a = period_avgs[k]
    b = period_avgs[k-1]

    ru = rel_l2_vec(a["uphi_bar"], a["ur_bar"], b["uphi_bar"], b["ur_bar"])
    rw = rel_l2(a["w_bar"], b["w_bar"])
    rp = rel_l2(a["p_bar"], b["p_bar"])

    print(
        f"avg period {k} vs {k-1}: "
        f"[{a['t0']:.6f},{a['t1']:.6f}] vs [{b['t0']:.6f},{b['t1']:.6f}]  "
        f"relL2(ubar)={ru:.3e}, relL2(wbar)={rw:.3e}, relL2(pbar)={rp:.3e}"
    )

    print("\nMulti-period averaged comparisons:")
print(f"Using avg_periods = {avg_periods}")

multi_avgs = []

for k in range(len(period_avgs) - avg_periods + 1):
    block = period_avgs[k:k+avg_periods]

    uphi_bar = np.mean([b["uphi_bar"] for b in block], axis=0)
    ur_bar   = np.mean([b["ur_bar"]   for b in block], axis=0)
    p_bar    = np.mean([b["p_bar"]    for b in block], axis=0)
    w_bar    = np.mean([b["w_bar"]    for b in block], axis=0)

    multi_avgs.append({
        "k0": k,
        "k1": k + avg_periods - 1,
        "t0": block[0]["t0"],
        "t1": block[-1]["t1"],
        "uphi_bar": uphi_bar,
        "ur_bar": ur_bar,
        "p_bar": p_bar,
        "w_bar": w_bar,
    })

for k in range(1, len(multi_avgs)):
    a = multi_avgs[k]
    b = multi_avgs[k-1]

    ru = rel_l2_vec(a["uphi_bar"], a["ur_bar"], b["uphi_bar"], b["ur_bar"])
    rw = rel_l2(a["w_bar"], b["w_bar"])
    rp = rel_l2(a["p_bar"], b["p_bar"])

    print(
        f"avg periods {a['k0']}-{a['k1']} vs {b['k0']}-{b['k1']}: "
        f"[{a['t0']:.6f},{a['t1']:.6f}] vs [{b['t0']:.6f},{b['t1']:.6f}]  "
        f"relL2(ubar)={ru:.3e}, relL2(wbar)={rw:.3e}, relL2(pbar)={rp:.3e}"
    )