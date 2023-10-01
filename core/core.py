import casadi as ca
import numpy as np

class mpc():
    def __init__(self, mpc_params):

        N = mpc_params['N']
        Q = mpc_params['Q']
        R = mpc_params['R']
        P = mpc_params['P']

        opti = ca.Opti()

        X = opti.variable(8, N+1)
        U = opti.variable(2, N)
        X0 = opti.variable(8)
        Xr = opti.variable(8, N)

        obj = 0.0

        for i in range(len(N)):
            obj += np.transpose(X[:, i+1] - Xr[:, i]) * Q * (X[:, i] - Xr[:, i]) + np.transpose(U[:, i]) * R * U[:, i]
            opti.subject_to(X[:, i+1] == f(X[:, i], U[:, i]))

        