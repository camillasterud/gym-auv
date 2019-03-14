import numpy as np

M = 18.82
I_zz = 1.77
X_udot = 0.421
Y_vdot = -27.2
Y_rdot = -1.83
N_rdot = -4.34
X_uu = -3.11
Y_vv = -3.01
N_rr = -2
Y_rr = 0.632
N_vv = -3.18
N_uudr = -6.08

THRUST_MIN_AUV = 0
THRUST_MAX_AUV = 10
RUDDER_MAX_AUV = 0.1
MAX_SPEED = 1.8

M_RB = np.diag([M, M, I_zz])

M_A = -np.array([
    [X_udot, 0, 0],
    [0, Y_vdot, Y_rdot],
    [0, Y_rdot, N_rdot]
    ])

M_inv = np.linalg.inv(M_RB + M_A)

D_quad = -np.diag([X_uu, Y_vv, N_rr])

def D(u, v, r):
    return D_quad @ np.diag(np.reshape(np.absolute([u, v, r]), (3,)))

def B(u):
    return np.array([
        [1, 0],
        [0, 0],
        [0, N_uudr*u*u],
    ])
