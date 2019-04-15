import numpy as np

m = 18
I_zz = 0
X_udot = -1
Y_vdot = -16
N_rdot = -2.1
X_u = -2.4
Y_v = -23
Y_r = 11.5
N_v = -3.1
N_r = -9.7
X_uu = -2.4
Y_vv = -80
Y_rr = 0.3
N_vv = -1.5
N_rr = -9.1
Y_uvb = -0.5*1000*np.pi*1.24*(0.15/2)**2
Y_uvf = -1000*3*0.0064
Y_urf = -0.4*Y_uvf
N_uvb = (-0.65*1.08 + 0.4)*Y_uvb
N_uvf = -0.4*Y_uvf
N_urf = -0.4*Y_urf
Y_uudr = 19.2
N_uudr = -0.4*Y_uudr


THRUST_MIN_AUV = 0
THRUST_MAX_AUV = 14.0417
RUDDER_MAX_AUV = 25*2*np.pi/360
MAX_SPEED = 2

M_RB = np.diag([m, m, I_zz])

M_A = -np.diag([X_udot, Y_vdot,N_rdot])

M_inv = np.linalg.inv(M_RB + M_A)

C_RB = np.array([[0, 0, -m],
                 [0, 0, m],
                 [m, -m, 0]])
C_A = np.array([[0, 0, Y_vdot],
                [0, 0, -X_udot],
                [-Y_vdot, X_udot, 0]])
D_lin = -np.array([[X_u, 0, 0],
                   [0, Y_v, Y_r],
                   [0, N_v, N_r]])

D_quad = -np.diag([X_uu, Y_vv, N_rr])

L_mat = np.array([[0, 0, 0],
                  [0, 30, -7.7],
                  [0, -9.9, 3.1]])

def D(nu):
    D = (D_lin
         + D_quad @ np.diag(np.absolute(nu)))
    D[2, 1] += D_quad[2, 1]*abs(nu[1])
    D[1, 2] += D_quad[1, 2]*abs(nu[2])
    return D

def B(nu):
    return np.array([
        [1, 0],
        [0, Y_uudr*(nu[0]**2)],
        [0, N_uudr*(nu[0]**2)],
    ])

def C(nu):
    C = C_RB + C_A
    C[0,2] = C[0,2]*nu[1]
    C[1,2] = C[1,2]*nu[0]
    C[2,0] = C[2,0]*nu[1]
    C[2,1] = C[2,1]*nu[0]
    return C

def L(nu):
    return L_mat*nu[0]
