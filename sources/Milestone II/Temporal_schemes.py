import numpy as np
from scipy.optimize import newton


def Euler(U, t, dt, F):

    return U + dt * F(U, t)

def RK4(U, t, dt, F):

        k1 = dt * F(U, t)
        k2 = dt * F(U + k1/2, t + dt/2)
        k3 = dt * F(U + k2/2, t + dt/2)
        k4 = dt * F(U + k3, t + dt)

        k = (k1 + 2*k2 + 2*k3 + k4)/6

        return U + k

def Euler_Inv(U, t, dt, F):

    def Residual_EI(X):

        return X - U - dt*F(X, t)

    return newton(Residual_EI, U, tol = 1e-05)

def Crank_Nicolson(U, t, dt, F):

    def Residual_CN(X):

        return X - U - dt * (F(X, t+dt) + F(U, t)) / 2

    return newton(Residual_CN, x0 = U, tol = 1e-05)