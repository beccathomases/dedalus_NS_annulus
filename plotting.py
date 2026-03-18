import numpy as np
import h5py
import matplotlib.pyplot as plt

fname = "Re40_40per_dt01/snapshots/snapshots_s9.h5"

Ri, Ro = 1.0, 8.0   # your run parameters

with h5py.File(fname, "r") as f:
    print("Available tasks:", list(f["tasks"].keys()))
    sim_time = np.array(f["scales/sim_time"])
    print("Saved times:", sim_time)

    k = -1   # last saved snapshot
    print("Plotting snapshot at t =", sim_time[k])

    udata = np.array(f["tasks/u"][k])
    p     = np.array(f["tasks/p"][k]).squeeze()
    vort  = np.array(f["tasks/vorticity"][k]).squeeze()
    divu  = np.array(f["tasks/divu"][k]).squeeze()

# figure out component axis for u
if udata.ndim != 3:
    raise ValueError(f"Expected 3D u snapshot after time indexing, got shape {udata.shape}")

if udata.shape[0] == 2:
    # shape = (2, nphi, nr)
    uphi = udata[0]
    ur   = udata[1]
elif udata.shape[-1] == 2:
    # shape = (nphi, nr, 2)
    uphi = udata[..., 0]
    ur   = udata[..., 1]
else:
    raise ValueError(f"Could not identify component axis in u with shape {udata.shape}")

nphi, nr = uphi.shape

# build coordinates from known geometry
phi = np.linspace(0, 2*np.pi, nphi, endpoint=False)
r   = np.linspace(Ri, Ro, nr)

PHI, R = np.meshgrid(phi, r, indexing="ij")
X = R * np.cos(PHI)
Y = R * np.sin(PHI)

# convert to Cartesian velocity for plotting
ux = ur*np.cos(PHI) - uphi*np.sin(PHI)
uy = ur*np.sin(PHI) + uphi*np.cos(PHI)

# scalar plots
fig, axs = plt.subplots(2, 2, figsize=(12, 10))

pcm = axs[0,0].pcolormesh(X, Y, p, shading="auto")
axs[0,0].set_title("pressure")
axs[0,0].set_aspect("equal")
fig.colorbar(pcm, ax=axs[0,0])

pcm = axs[0,1].pcolormesh(X, Y, vort, shading="auto")
axs[0,1].set_title("vorticity")
axs[0,1].set_aspect("equal")
fig.colorbar(pcm, ax=axs[0,1])

pcm = axs[1,0].pcolormesh(X, Y, divu, shading="auto")
axs[1,0].set_title("divu")
axs[1,0].set_aspect("equal")
fig.colorbar(pcm, ax=axs[1,0])

speed = np.sqrt(ux**2 + uy**2)
pcm = axs[1,1].pcolormesh(X, Y, speed, shading="auto")
axs[1,1].set_title("|u|")
axs[1,1].set_aspect("equal")
fig.colorbar(pcm, ax=axs[1,1])

plt.tight_layout()
plt.show()