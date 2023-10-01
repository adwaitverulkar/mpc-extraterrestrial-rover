import casadi as ca
import numpy as np
from core.helpers import sigmoid, lateral_force
import sys
import os

class rover_model_nlbicycle():
    def __init__(self, rover_params):

        x = ca.SX.sym('x', 8)
        u = ca.SX.sym('u', 2)
    
        M = rover_params['M']
        Iz = rover_params['Iz']

        lf = rover_params['lf']
        lr = rover_params['lr']

        L = lf+lr
        
        h = rover_params['h']

        eps = rover_params['eps']

        alphaf = x[6]-ca.atan2(x[4] + lf*x[5], ca.sqrt(x[3]*x[3]+eps*eps))
        alphar = -ca.atan2(x[4] - lr*x[5], ca.sqrt(x[3]*x[3]+eps*eps))
        
        acc_sl = rover_params['acc_sl']
        brake_sl = rover_params['brake_sl']

        Fx = sigmoid(x[7])*acc_sl*x[7] + sigmoid(-x[7])*brake_sl*x[7]

        w_tr = Fx*h/L

        Fzf = M*9.81*lr/L - w_tr
        Fzr = M*9.81*lf/L + w_tr

        Bf = rover_params['Bf']
        Cf = rover_params['Cf']
        Df = rover_params['Df']
        Ef = rover_params['Ef']

        Br = rover_params['Br']
        Cr = rover_params['Cr']
        Dr = rover_params['Dr']
        Er = rover_params['Er']

        Ffy = lateral_force(Fzf, alphaf, Bf, Cf, Df, Ef)
        Fry = lateral_force(Fzr, alphar, Br, Cr, Dr, Er)

        ode_rhs = ca.vertcat(x[3]*ca.cos(x[2]) - x[4]*ca.sin(x[2]),\
                            x[3]*ca.sin(x[2]) + x[4]*ca.sin(x[2]),\
                            x[5],\
                            1/M*(Fx-Ffy*ca.sin(x[6])+M*x[4]*x[5]),\
                            1/M*(Fry+Ffy*ca.cos(x[6])+M*x[3]*x[5]),\
                            1/Iz*(Ffy*ca.cos(x[6])*lf-Fry*lr),\
                            u[0],\
                            u[1])
        
        self.f = ca.Function('f', [x, u], [ode_rhs])

        Ts = rover_params['Ts']

        self.model_ss = ca.Function('model_ss', [x, u], [x+self.f(x, u)*Ts]).expand()
        

        k1 = self.f(x, u)
        k2 = self.f(x+Ts*k1/2.0, u)
        k3 = self.f(x+Ts*k2/2.0, u)
        k4 = self.f(x+Ts*k3, u)

        self.model_ms = ca.Function('model_ms', [x, u], [x+Ts/6.0*(k1 + 2.0*k2 + 2.0*k3 + k4)]).expand()
        
    def ss_model(self):
        return self.model_ss
    
    def ms_model(self):
        return self.model_ms

