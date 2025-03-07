import numpy as np


def computeADE(traj1, traj2):
    traj1 = traj1[:traj2.shape[0],:]
    distances = np.linalg.norm((traj1-traj2), axis = 1)
    ade = np.mean(distances)
    return float(ade)


def computeFDE(traj1, traj2):
    t = traj2.shape[0]
    return float(np.linalg.norm(traj1[t-1,:] - traj2[t-1,:]))
