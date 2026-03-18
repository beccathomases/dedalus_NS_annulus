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
snap_dir = "Re40_caseA/seg01_from_rest_snap0p1/snapshots"

forcing_period = 1.0
refined_n_periods = 3
snapshot_dt = 0.01              # save interval used in this run
steps_per_period = int(round(forcing_period / snapshot_dt))
avg_periods = 2                 # 1 = single-period compare, 2 or 3 = sliding multi-period average

# Geometry / parameters used in the run
Ri = 1.0
Ro = 8.0
Re_target = 40.0
Omega_ref = 2*np.pi
Rref = 1.0
nu = (Rref**2) * Omega_ref / Re_target

# ------------------------------------------------------------
# Window for convergence-of-streaming section
# If None, use all available data
# ------------------------------------------------------------
conv_t_start = None
conv_t_end = None

# ------------------------------------------------------------
# Window for refined-window analysis (force/streamlines)
# If use_last_period=True, ignore refined_t_start/refined_t_end
# ------------------------------------------------------------
refined_use_last_period = True
refined_t_start = None
refined_t_end = None

# Cartesian grid for period-averaged streamline plot
cart_N = 300

# Outputs
conv_csv = os.path.join(snap_dir, "streaming_convergence.csv")
force_csv = os.path.join(snap_dir, "refined_force_timeseries.csv")
summary_txt = os.path.join(snap_dir, "combined_analysis_summary.txt")
diag_fig = os.path.join(snap_dir, "combined_diagnostics.png")
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

    for name, arr in cands:
        lname = name.lower()
        if "phi" in lname and arr.size == nphi:
            phi = arr.copy()
        if lname.endswith("/r") and arr.size == nr:
            r = arr.copy()

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
            spans = [abs((arr.max() - arr.min()) - (Ro - Ri)) for arr in r_opts]
            r = r_opts[int(np.argmin(spans))].copy()

    if phi is None:
        phi = np.linspace(0, 2*np.pi, nphi, endpoint=False)

    if r is None:
        raise RuntimeError("Could not determine radial coordinate array from HDF5 file.")

    phi = np.array(phi).squeeze()
    r = np.array(r).squeeze()

    if np.any(np.diff(phi) < 0):
        idx = np.argsort(phi)
        phi = phi[idx]
    if np.any(np.diff(r) < 0):
        idx = np.argsort(r)
        r = r[idx]

    return phi, r

def periodic_phi_derivative(q, phi):
    dphi = phi[1] - phi[0]
    return (np.roll(q, -1, axis=0) - np.roll(q, 1, axis=0)) / (2*dphi)

def radial_derivative(q, r):
    return np.gradient(q, r, axis=1, edge_order=2)

def cylinder_force_from_snapshot(uphi, ur, p, phi, r, Ri, nu):
    """
    Net force on inner cylinder from one snapshot.
    """
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

def periodic_extension(phi, q):
    phi_ext = np.concatenate([phi, [phi[0] + 2*np.pi]])
    q_ext = np.concatenate([q, q[0:1, :]], axis=0)
    return phi_ext, q_ext

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

    # remove duplicate times, which can happen at restart joins
    deduped = []
    last_t = None
    for rec in records:
        t = rec[0]
        if last_t is None or abs(t - last_t) > 1e-12:
            deduped.append(rec)
            last_t = t

    return deduped

def select_records(records, t_start=None, t_end=None):
    if t_start is None and t_end is None:
        return records[:]
    tol = 1e-12
    out = []
    for rec in records:
        t = rec[0]
        if (t_start is None or t >= t_start - tol) and (t_end is None or t <= t_end + tol):
            out.append(rec)
    return out

def read_snapshot(fname, j):
    with h5py.File(fname, "r") as f:
        u_snap = np.array(f["tasks/u"][j])
        p_snap = np.array(f["tasks/p"][j]).squeeze()
        om_snap = np.array(f["tasks/vorticity"][j]).squeeze()
    uphi, ur = get_u_components(u_snap)
    return uphi, ur, p_snap, om_snap

# ============================================================
# Load all snapshot records
# ============================================================
records = load_records(snap_dir)
all_times = np.array([rec[0] for rec in records])

print("Files being used:")
for f in sorted(set(rec[1] for rec in records), key=snapshot_file_key):
    print(" ", f)

print()
print(f"Number of snapshots: {len(records)}")
print(f"First saved time: {all_times[0]:.6f}")
print(f"Last saved time:  {all_times[-1]:.6f}")

# ============================================================
# Coordinates from first record
# ============================================================
with h5py.File(records[0][1], "r") as f:
    u0 = np.array(f["tasks/u"][records[0][2]])
    uphi0, ur0 = get_u_components(u0)
    nphi, nr = uphi0.shape
    phi, r = guess_phi_r_from_file(f, nphi, nr, Ri, Ro)

# ============================================================
# Part A: convergence of streaming / period-averaged fields
# ============================================================
conv_records = select_records(records, conv_t_start, conv_t_end)
conv_times = np.array([rec[0] for rec in conv_records])

print()
print("Convergence window:")
if conv_t_start is None and conv_t_end is None:
    print("  using all available snapshots")
else:
    print(f"  using [{conv_times[0]:.6f}, {conv_times[-1]:.6f}]")

n_periods = len(conv_records) // steps_per_period
leftover = len(conv_records) - n_periods * steps_per_period

print(f"steps_per_period = {steps_per_period}")
print(f"n_periods from integer chunking = {n_periods}")
print(f"leftover snapshots = {leftover}")

period_avgs = []

for k in range(n_periods):
    i0 = k * steps_per_period
    i1 = (k + 1) * steps_per_period

    uphi_sum = None
    ur_sum = None
    p_sum = None
    w_sum = None

    for j in range(i0, i1):
        t, fname, idx = conv_records[j]
        uphi, ur, p_snap, w_snap = read_snapshot(fname, idx)

        if uphi_sum is None:
            uphi_sum = np.zeros_like(uphi)
            ur_sum = np.zeros_like(ur)
            p_sum = np.zeros_like(p_snap)
            w_sum = np.zeros_like(w_snap)

        uphi_sum += uphi
        ur_sum += ur
        p_sum += p_snap
        w_sum += w_snap

    period_avgs.append({
        "k": k,
        "t0": conv_records[i0][0],
        "t1": conv_records[i1 - 1][0],
        "uphi_bar": uphi_sum / steps_per_period,
        "ur_bar": ur_sum / steps_per_period,
        "p_bar": p_sum / steps_per_period,
        "w_bar": w_sum / steps_per_period,
    })

    print(
        f"Period chunk {k}: indices [{i0}:{i1-1}]  "
        f"times [{conv_records[i0][0]:.6f}, {conv_records[i1-1][0]:.6f}]  "
        f"span={conv_records[i1-1][0] - conv_records[i0][0]:.6f}"
    )

multi_avgs = []
for k in range(len(period_avgs) - avg_periods + 1):
    block = period_avgs[k:k + avg_periods]
    multi_avgs.append({
        "k0": block[0]["k"],
        "k1": block[-1]["k"],
        "t0": block[0]["t0"],
        "t1": block[-1]["t1"],
        "uphi_bar": np.mean([b["uphi_bar"] for b in block], axis=0),
        "ur_bar": np.mean([b["ur_bar"] for b in block], axis=0),
        "p_bar": np.mean([b["p_bar"] for b in block], axis=0),
        "w_bar": np.mean([b["w_bar"] for b in block], axis=0),
    })

conv_rows = []
for k in range(1, len(multi_avgs)):
    a = multi_avgs[k]
    b = multi_avgs[k - 1]

    ru = rel_l2_vec(a["uphi_bar"], a["ur_bar"], b["uphi_bar"], b["ur_bar"])
    rw = rel_l2(a["w_bar"], b["w_bar"])
    rp = rel_l2(a["p_bar"], b["p_bar"])

    conv_rows.append({
        "k0_new": a["k0"],
        "k1_new": a["k1"],
        "k0_old": b["k0"],
        "k1_old": b["k1"],
        "t0_new": a["t0"],
        "t1_new": a["t1"],
        "t0_old": b["t0"],
        "t1_old": b["t1"],
        "relL2_ubar": ru,
        "relL2_wbar": rw,
        "relL2_pbar": rp,
        "t_plot": a["t1"],
    })

print("\nStreaming convergence comparisons:")
print(f"Using avg_periods = {avg_periods}")
for row in conv_rows:
    print(
        f"avg periods {row['k0_new']}-{row['k1_new']} vs {row['k0_old']}-{row['k1_old']}: "
        f"[{row['t0_new']:.6f},{row['t1_new']:.6f}] vs "
        f"[{row['t0_old']:.6f},{row['t1_old']:.6f}]  "
        f"relL2(ubar)={row['relL2_ubar']:.3e}, "
        f"relL2(wbar)={row['relL2_wbar']:.3e}, "
        f"relL2(pbar)={row['relL2_pbar']:.3e}"
    )

if conv_rows:
    with open(conv_csv, "w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(conv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(conv_rows)

# ============================================================
# Part B: refined-window analysis for force + streamlines
# ============================================================

if refined_use_last_period:
    refined_t_end = all_times[-1]
    refined_t_start = refined_t_end - refined_n_periods * forcing_period

else:
    if refined_t_start is None or refined_t_end is None:
        raise ValueError("Set refined_t_start and refined_t_end, or use refined_use_last_period=True.")

selected = select_records(records, refined_t_start, refined_t_end)
if len(selected) < 2:
    raise RuntimeError("Not enough snapshots in selected refined window.")

print()
print(f"Refined analysis window [{refined_t_start:.6f}, {refined_t_end:.6f}]")
print(f"Number of selected snapshots: {len(selected)}")

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
    uphi, ur, p_snap, om_snap = read_snapshot(fname, j)
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
            int_ur = 0.5 * dt * (prev["ur"] + cur["ur"])
            int_p = 0.5 * dt * (prev["p"] + cur["p"])
            int_omega = 0.5 * dt * (prev["omega"] + cur["omega"])
        else:
            int_uphi += 0.5 * dt * (prev["uphi"] + cur["uphi"])
            int_ur += 0.5 * dt * (prev["ur"] + cur["ur"])
            int_p += 0.5 * dt * (prev["p"] + cur["p"])
            int_omega += 0.5 * dt * (prev["omega"] + cur["omega"])

        int_Fx += 0.5 * dt * (prev["Fx"] + cur["Fx"])
        int_Fy += 0.5 * dt * (prev["Fy"] + cur["Fy"])

    prev = cur

last = prev
T_used = last["t"] - first["t"]
if T_used <= 0:
    raise RuntimeError("Selected refined window has zero or negative duration.")

avg_uphi = int_uphi / T_used
avg_ur = int_ur / T_used
avg_p = int_p / T_used
avg_omega = int_omega / T_used
avg_Fx = int_Fx / T_used
avg_Fy = int_Fy / T_used
avg_Fmag = np.sqrt(avg_Fx**2 + avg_Fy**2)

rel_u = rel_l2_vec(last["uphi"], last["ur"], first["uphi"], first["ur"])
rel_p = rel_l2(last["p"], first["p"])
rel_omega = rel_l2(last["omega"], first["omega"])

with open(force_csv, "w", newline="") as fp:
    writer = csv.DictWriter(fp, fieldnames=list(force_rows[0].keys()))
    writer.writeheader()
    writer.writerows(force_rows)

with open(summary_txt, "w") as fp:
    fp.write("=== Streaming convergence section ===\n")
    fp.write(f"Convergence window: [{conv_records[0][0]:.12f}, {conv_records[-1][0]:.12f}]\n")
    fp.write(f"steps_per_period = {steps_per_period}\n")
    fp.write(f"avg_periods = {avg_periods}\n")
    fp.write(f"n_periods = {n_periods}\n")
    fp.write(f"leftover snapshots = {leftover}\n\n")

    if conv_rows:
        last_conv = conv_rows[-1]
        fp.write("Last streaming convergence comparison:\n")
        fp.write(
            f"  avg periods {last_conv['k0_new']}-{last_conv['k1_new']} "
            f"vs {last_conv['k0_old']}-{last_conv['k1_old']}\n"
        )
        fp.write(f"  relL2_ubar = {last_conv['relL2_ubar']:.6e}\n")
        fp.write(f"  relL2_wbar = {last_conv['relL2_wbar']:.6e}\n")
        fp.write(f"  relL2_pbar = {last_conv['relL2_pbar']:.6e}\n\n")

    fp.write("=== Refined-window section ===\n")
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

print("\nRefined-window summary:")
print(f"Window used: t = {first['t']:.6f} to {last['t']:.6f}")
print(f"relL2_u(start,end)      = {rel_u:.6e}")
print(f"relL2_p(start,end)      = {rel_p:.6e}")
print(f"relL2_omega(start,end)  = {rel_omega:.6e}")
print(f"<Fx> = {avg_Fx:.6e}, <Fy> = {avg_Fy:.6e}, |<F>| = {avg_Fmag:.6e}")
print(f"Saved convergence CSV to: {conv_csv}")
print(f"Saved force CSV to: {force_csv}")
print(f"Saved text summary to: {summary_txt}")

# ============================================================
# Figure 1: combined diagnostics
# ============================================================
fig, axs = plt.subplots(3, 1, figsize=(9, 12))

# top: convergence of streaming
if conv_rows:
    t_conv = np.array([row["t_plot"] for row in conv_rows])
    rel_ubar = np.array([row["relL2_ubar"] for row in conv_rows])
    rel_wbar = np.array([row["relL2_wbar"] for row in conv_rows])
    rel_pbar = np.array([row["relL2_pbar"] for row in conv_rows])

    axs[0].semilogy(t_conv, rel_ubar, marker='o', label='relL2(ubar)')
    axs[0].semilogy(t_conv, rel_wbar, marker='o', label='relL2(wbar)')
    axs[0].semilogy(t_conv, rel_pbar, marker='o', label='relL2(pbar)')
    axs[0].set_title(f"Streaming convergence (avg_periods = {avg_periods})")
    axs[0].set_xlabel("end time of new averaged block")
    axs[0].set_ylabel("relative change")
    axs[0].grid(True)
    axs[0].legend()
else:
    axs[0].text(0.5, 0.5, "Not enough periods for convergence plot", ha='center', va='center')
    axs[0].set_axis_off()

# middle: Fx and Fy
tvals = np.array([row["t"] for row in force_rows])
Fxvals = np.array([row["Fx"] for row in force_rows])
Fyvals = np.array([row["Fy"] for row in force_rows])
Fmagvals = np.array([row["Fmag"] for row in force_rows])

axs[1].plot(tvals, Fxvals, marker='o', label='Fx')
axs[1].plot(tvals, Fyvals, marker='o', label='Fy')
axs[1].axhline(avg_Fx, linestyle='--', linewidth=1, label='<Fx>')
axs[1].axhline(avg_Fy, linestyle='--', linewidth=1, label='<Fy>')
axs[1].set_title("Net force on inner cylinder over refined window")
axs[1].set_xlabel("t")
axs[1].set_ylabel("force")
axs[1].grid(True)
axs[1].legend()

# bottom: |F|
axs[2].plot(tvals, Fmagvals, marker='o', label='|F|')
axs[2].axhline(avg_Fmag, linestyle='--', linewidth=1, label='|<F>|')
axs[2].set_title("Force magnitude over refined window")
axs[2].set_xlabel("t")
axs[2].set_ylabel("|force|")
axs[2].grid(True)
axs[2].legend()

plt.tight_layout()
plt.savefig(diag_fig, dpi=150)
plt.show()
print(f"Saved combined diagnostics figure to: {diag_fig}")

# ============================================================
# Figure 2: period-averaged flow streamlines
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
plt.colorbar(cf, ax=ax, label="|<u>| over refined window")
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

zoom_stream_fig = os.path.join(snap_dir, "period_averaged_streamlines_zoom.png")

fig3, axz = plt.subplots(figsize=(8, 8))
cfz = axz.contourf(X, Y, Speed, levels=80)
plt.colorbar(cfz, ax=axz, label="|<u>| over refined window")

axz.streamplot(
    xg, yg, Ux, Uy,
    density=5.0,
    linewidth=0.8,
    arrowsize=0.8
)

inner = plt.Circle((0, 0), Ri, color='white', zorder=10)
outer = plt.Circle((0, 0), Ro, fill=False, color='k', linewidth=1.0)
axz.add_patch(inner)
axz.add_patch(outer)

zoom_R = 4.0
axz.set_aspect('equal')
axz.set_xlim(-zoom_R, zoom_R)
axz.set_ylim(-zoom_R, zoom_R)
axz.set_title("Streamlines of period-averaged flow (zoom)")
axz.set_xlabel("x")
axz.set_ylabel("y")

plt.tight_layout()
plt.savefig(zoom_stream_fig, dpi=150)
plt.show()

print(f"Saved zoomed streamline figure to: {zoom_stream_fig}")