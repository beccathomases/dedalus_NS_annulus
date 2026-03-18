import glob
import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
from scipy.io import loadmat


# ============================================================
# CONFIG
# ============================================================
MATLAB_FILE = "myData.mat"
SNAPSHOT_GLOB = "Re40_caseA/seg02_refine_snap0p01/snapshots/*.h5"

PERIOD = 1.0          # forcing period
MAKE_PLOTS = True

# Optional constant angular shift, if needed later
THETA_SHIFT = 0.0


# ============================================================
# HELPERS
# ============================================================
def fro_rel_err(a, b):
    num = np.linalg.norm(a - b)
    den = np.linalg.norm(b)
    return num if den == 0 else num / den


def load_mydata_mat(fname):
    needed = ["streamFunction", "ur", "ut", "ux", "uy", "x", "y"]
    data = loadmat(fname)
    out = {}
    for k in needed:
        if k not in data:
            raise KeyError(f"Field '{k}' not found in {fname}")
        out[k] = np.array(data[k], dtype=float)
    print(f"Loaded {fname} with scipy.io.loadmat")
    return out


def structure_score(x, y):
    r = np.sqrt(x**2 + y**2)
    th = np.mod(np.arctan2(y, x), 2*np.pi)
    return np.max(np.std(r, axis=1)) + np.max(np.std(th, axis=0))


def maybe_fix_orientation(data):
    x = data["x"]
    y = data["y"]

    score_native = structure_score(x, y)
    score_T = structure_score(x.T, y.T)

    if score_T < score_native:
        print("Transposing collaborator arrays so rows=radius, cols=angle")
        for k in data:
            data[k] = data[k].T.copy()
    else:
        print("Keeping collaborator arrays as loaded")
    return data


def sort_collab_grid(data):
    x = data["x"]
    y = data["y"]

    r = np.sqrt(x**2 + y**2)
    th = np.mod(np.arctan2(y, x), 2*np.pi)

    row_order = np.argsort(np.mean(r, axis=1))
    col_order = np.argsort(np.mean(th, axis=0))

    for k in data:
        data[k] = data[k][row_order][:, col_order]

    x = data["x"]
    y = data["y"]
    r = np.sqrt(x**2 + y**2)
    th = np.mod(np.arctan2(y, x), 2*np.pi)

    rvec = np.mean(r, axis=1)
    thvec = np.mean(th, axis=0)

    return data, rvec, thvec, r, th


def load_all_snapshots(snapshot_glob):
    files = sorted(glob.glob(snapshot_glob))
    if not files:
        raise FileNotFoundError(f"No files matched {snapshot_glob}")

    all_t = []
    all_u = []
    phi = None
    r = None

    for fname in files:
        with h5py.File(fname, "r") as f:
            times = np.array(f["scales/sim_time"])
            u = np.array(f["tasks/u"])   # (nt, 2, Nphi, Nr)

            scales_keys = list(f["scales"].keys())
            phi_key = [s for s in scales_keys if s.startswith("phi_hash_")][0]
            r_key   = [s for s in scales_keys if s.startswith("r_hash_")][0]

            phi_here = np.array(f["scales"][phi_key]).squeeze()
            r_here   = np.array(f["scales"][r_key]).squeeze()

            if phi is None:
                phi = phi_here
                r = r_here
            else:
                if not (np.allclose(phi, phi_here) and np.allclose(r, r_here)):
                    raise ValueError(f"Grid mismatch across snapshot files, including {fname}")

            all_t.append(times)
            all_u.append(u)

    t = np.concatenate(all_t, axis=0)
    u = np.concatenate(all_u, axis=0)

    # sort by time
    order = np.argsort(t)
    t = t[order]
    u = u[order]

    # drop duplicate times if any
    keep = np.ones(len(t), dtype=bool)
    keep[1:] = np.diff(t) > 1e-12
    t = t[keep]
    u = u[keep]

    # sort phi and r
    phi_order = np.argsort(phi)
    r_order = np.argsort(r)

    phi = phi[phi_order]
    r = r[r_order]
    u = u[:, :, phi_order, :]
    u = u[:, :, :, r_order]

    return t, phi, r, u


def average_last_period(t, u, period=1.0):
    t_end = t[-1]
    t_start_target = t_end - period

    mask = (t >= t_start_target - 1e-12) & (t <= t_end + 1e-12)
    t_sel = t[mask]
    u_sel = u[mask]

    if len(t_sel) < 2:
        raise ValueError("Not enough snapshots in the last period to average.")

    # time-average via trapezoidal rule
    duration = t_sel[-1] - t_sel[0]
    u_mean = np.trapezoid(u_sel, x=t_sel, axis=0) / duration

    # fluctuation level around the mean
    fluct = u_sel - u_mean[None, ...]
    u_rms_fluct = np.sqrt(np.trapezoid(fluct**2, x=t_sel, axis=0) / duration)

    return t_sel, u_sel, u_mean, u_rms_fluct


def make_periodic_phi_interpolator(phi, r, q_phi_r):
    phi_ext = np.concatenate([phi, [phi[0] + 2*np.pi]])
    q_ext = np.concatenate([q_phi_r, q_phi_r[0:1, :]], axis=0)

    return RegularGridInterpolator(
        (phi_ext, r),
        q_ext,
        method="linear",
        bounds_error=False,
        fill_value=np.nan,
    )


def summarize(name, a):
    print(f"{name:10s} min={np.nanmin(a): .6e}  max={np.nanmax(a): .6e}  norm={np.linalg.norm(a): .6e}")


# ============================================================
# LOAD COLLABORATOR DATA
# ============================================================
collab = load_mydata_mat(MATLAB_FILE)
collab = maybe_fix_orientation(collab)
collab, rC_vec, thC_vec, rC_grid, thC_grid = sort_collab_grid(collab)

ur_C = collab["ur"]
ut_C = collab["ut"]
ux_C = collab["ux"]
uy_C = collab["uy"]
x_C  = collab["x"]
y_C  = collab["y"]

print("\nCollaborator data shapes:")
for k, v in collab.items():
    print(f"  {k}: {v.shape}")

print("\nCollaborator grid checks:")
print("  max std(radius across rows)   =", np.max(np.std(rC_grid, axis=1)))
print("  max std(theta across cols)    =", np.max(np.std(thC_grid, axis=0)))


# ============================================================
# LOAD DEDALUS SNAPSHOTS AND AVERAGE OVER LAST PERIOD
# ============================================================
t_D, phi_D, r_D, u_D = load_all_snapshots(SNAPSHOT_GLOB)
t_sel, u_sel, u_mean, u_rms_fluct = average_last_period(t_D, u_D, period=PERIOD)

print("\nDedalus snapshot collection:")
print("  number of times total  :", len(t_D))
print("  time range             :", t_D[0], "to", t_D[-1])
print("  number in last period  :", len(t_sel))
print("  averaging window       :", t_sel[0], "to", t_sel[-1])
print("  phi shape              :", phi_D.shape)
print("  r shape                :", r_D.shape)
print("  u_mean shape           :", u_mean.shape)

# From your code: component 0 = u_phi, component 1 = u_r
uphi_mean_D = u_mean[0]   # (phi,r)
ur_mean_D   = u_mean[1]   # (phi,r)

uphi_rms_D = u_rms_fluct[0]
ur_rms_D   = u_rms_fluct[1]

# quick fluctuation diagnostics
mean_speed_D = np.sqrt(uphi_mean_D**2 + ur_mean_D**2)
rms_fluct_speed_D = np.sqrt(uphi_rms_D**2 + ur_rms_D**2)

print("\nDedalus mean/fluctuation summary on native grid:")
summarize("ur_mean", ur_mean_D)
summarize("ut_mean", uphi_mean_D)
summarize("fluct_r", ur_rms_D)
summarize("fluct_t", uphi_rms_D)
print("  fluct/mean speed ratio =", np.linalg.norm(rms_fluct_speed_D) / np.linalg.norm(mean_speed_D))


# ============================================================
# INTERPOLATE DEDALUS MEAN FIELD ONTO COLLABORATOR GRID
# ============================================================
th_query = np.mod(thC_grid + THETA_SHIFT, 2*np.pi)

interp_ur = make_periodic_phi_interpolator(phi_D, r_D, ur_mean_D)
interp_ut = make_periodic_phi_interpolator(phi_D, r_D, uphi_mean_D)

pts = np.column_stack([th_query.ravel(), rC_grid.ravel()])

ur_D_on_C = interp_ur(pts).reshape(rC_grid.shape)
ut_D_on_C = interp_ut(pts).reshape(rC_grid.shape)

ux_D_on_C = ur_D_on_C * np.cos(thC_grid) - ut_D_on_C * np.sin(thC_grid)
uy_D_on_C = ur_D_on_C * np.sin(thC_grid) + ut_D_on_C * np.cos(thC_grid)


# ============================================================
# ERRORS
# ============================================================
print("\nRelative errors using Dedalus last-period mean:")
print(f"  ur : {fro_rel_err(ur_D_on_C, ur_C):.6e}")
print(f"  ut : {fro_rel_err(ut_D_on_C, ut_C):.6e}")
print(f"  ux : {fro_rel_err(ux_D_on_C, ux_C):.6e}")
print(f"  uy : {fro_rel_err(uy_D_on_C, uy_C):.6e}")

print("\nMax abs errors:")
print(f"  ur : {np.nanmax(np.abs(ur_D_on_C - ur_C)):.6e}")
print(f"  ut : {np.nanmax(np.abs(ut_D_on_C - ut_C)):.6e}")
print(f"  ux : {np.nanmax(np.abs(ux_D_on_C - ux_C)):.6e}")
print(f"  uy : {np.nanmax(np.abs(uy_D_on_C - uy_C)):.6e}")

print("\nRange summary:")
summarize("ur_C", ur_C)
summarize("ut_C", ut_C)
summarize("ux_C", ux_C)
summarize("uy_C", uy_C)
summarize("ur_D_cmp", ur_D_on_C)
summarize("ut_D_cmp", ut_D_on_C)
summarize("ux_D_cmp", ux_D_on_C)
summarize("uy_D_cmp", uy_D_on_C)


theta_xy = np.arctan2(y_C, x_C)

def re(a,b):
    return np.linalg.norm(a-b)/np.linalg.norm(b)

ux1 = ur_C*np.cos(theta_xy) - ut_C*np.sin(theta_xy)
uy1 = ur_C*np.sin(theta_xy) + ut_C*np.cos(theta_xy)

ux2 = ur_C*np.cos(theta_xy) + ut_C*np.sin(theta_xy)
uy2 = ur_C*np.sin(theta_xy) - ut_C*np.cos(theta_xy)

ux3 = ut_C*np.cos(theta_xy) - ur_C*np.sin(theta_xy)
uy3 = ut_C*np.sin(theta_xy) + ur_C*np.cos(theta_xy)

ux4 = ut_C*np.cos(theta_xy) + ur_C*np.sin(theta_xy)
uy4 = ut_C*np.sin(theta_xy) - ur_C*np.cos(theta_xy)

print("standard :", re(ux1,ux_C)+re(uy1,uy_C))
print("flip_ut  :", re(ux2,ux_C)+re(uy2,uy_C))
print("swap     :", re(ux3,ux_C)+re(uy3,uy_C))
print("swapflip :", re(ux4,ux_C)+re(uy4,uy_C))

rC = np.sqrt(x_C**2 + y_C**2)
print(rC.min(), rC.max())
print(r_D.min(), r_D.max())

# ============================================================
# PLOTS
# ============================================================
if MAKE_PLOTS:
    speed_C = np.sqrt(ux_C**2 + uy_C**2)
    speed_D = np.sqrt(ux_D_on_C**2 + uy_D_on_C**2)

    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)

    im = axes[0, 0].pcolormesh(thC_vec, rC_vec, ur_C, shading="auto")
    axes[0, 0].set_title("Collaborator ur")
    axes[0, 0].set_xlabel("theta")
    axes[0, 0].set_ylabel("r")
    fig.colorbar(im, ax=axes[0, 0])

    im = axes[0, 1].pcolormesh(thC_vec, rC_vec, ur_D_on_C, shading="auto")
    axes[0, 1].set_title("Dedalus mean ur on collaborator grid")
    axes[0, 1].set_xlabel("theta")
    axes[0, 1].set_ylabel("r")
    fig.colorbar(im, ax=axes[0, 1])

    im = axes[0, 2].pcolormesh(thC_vec, rC_vec, ur_D_on_C - ur_C, shading="auto")
    axes[0, 2].set_title("Difference ur")
    axes[0, 2].set_xlabel("theta")
    axes[0, 2].set_ylabel("r")
    fig.colorbar(im, ax=axes[0, 2])

    im = axes[1, 0].pcolormesh(thC_vec, rC_vec, ut_C, shading="auto")
    axes[1, 0].set_title("Collaborator ut")
    axes[1, 0].set_xlabel("theta")
    axes[1, 0].set_ylabel("r")
    fig.colorbar(im, ax=axes[1, 0])

    im = axes[1, 1].pcolormesh(thC_vec, rC_vec, ut_D_on_C, shading="auto")
    axes[1, 1].set_title("Dedalus mean ut on collaborator grid")
    axes[1, 1].set_xlabel("theta")
    axes[1, 1].set_ylabel("r")
    fig.colorbar(im, ax=axes[1, 1])

    im = axes[1, 2].pcolormesh(thC_vec, rC_vec, ut_D_on_C - ut_C, shading="auto")
    axes[1, 2].set_title("Difference ut")
    axes[1, 2].set_xlabel("theta")
    axes[1, 2].set_ylabel("r")
    fig.colorbar(im, ax=axes[1, 2])

    plt.show()

    plt.figure(figsize=(7, 4))
    speed_time = np.sqrt(u_sel[:, 0]**2 + u_sel[:, 1]**2)
    rms_speed_time = np.sqrt(np.mean(speed_time**2, axis=(1, 2)))
    plt.plot(t_sel, rms_speed_time, marker="o")
    plt.xlabel("time")
    plt.ylabel("domain RMS speed")
    plt.title("Dedalus speed over last period")
    plt.grid(True)
    plt.show()

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)

    im = axes[0].pcolormesh(thC_vec, rC_vec, speed_C, shading="auto")
    axes[0].set_title("Collaborator speed")
    axes[0].set_xlabel("theta")
    axes[0].set_ylabel("r")
    fig.colorbar(im, ax=axes[0])

    im = axes[1].pcolormesh(thC_vec, rC_vec, speed_D, shading="auto")
    axes[1].set_title("Dedalus mean speed on collaborator grid")
    axes[1].set_xlabel("theta")
    axes[1].set_ylabel("r")
    fig.colorbar(im, ax=axes[1])

    im = axes[2].pcolormesh(thC_vec, rC_vec, speed_D - speed_C, shading="auto")
    axes[2].set_title("Difference speed")
    axes[2].set_xlabel("theta")
    axes[2].set_ylabel("r")
    fig.colorbar(im, ax=axes[2])

    plt.show()