import h5py
import numpy as np

fname = "Re40_caseA/seg02_refine_snap0p01/snapshots/snapshots_s1.h5"

with h5py.File(fname, "r") as f:
    # find coordinate datasets
    phi_key = [k for k in f["scales"].keys() if k.startswith("phi_hash_")][0]
    r_key   = [k for k in f["scales"].keys() if k.startswith("r_hash_")][0]

    phi = np.array(f["scales"][phi_key])   # shape (128,)
    r   = np.array(f["scales"][r_key])     # shape (512,)

    # choose a time index, e.g. last snapshot
    k = -1
    u = np.array(f["tasks/u"][k])          # shape (2,128,512)

u_phi = u[0,:,:]      # (phi,r)
u_r   = u[1,:,:]      # (phi,r)

# transpose into collaborator ordering: (r,phi)
u_phi_rt = u_phi.T    # (512,128)
u_r_rt   = u_r.T


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