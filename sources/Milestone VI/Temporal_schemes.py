from tkinter import W
from numpy import cos, linspace, pi, sin, zeros, size
from mpmath import findroot
from scipy.optimize import newton

from Ext_RK_Emb import rk_scheme, butcher_arr, step_size


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

    return newton(Residual_EI, x0 = U, tol = 1e-05, maxiter = 10000)

def Crank_Nicolson(U, t, dt, F):

    def Residual_CN(X):

        return X - U - dt * (F(X, t+dt) + F(U, t)) / 2

    return newton(Residual_CN, x0 = U, tol = 1e-05, maxiter= 1000)

def Leap_Frog(U1, U2, dt, t, F):

    return U1 + 2*dt*F(U2,t)


def RK_emb(U, t, dt, F):
    
    RK_emb.__name__ =="Embedded RK"
    
    tol = 1e-10

    V1 = rk_scheme(1, U, t, dt, F) 
    V2 = rk_scheme(2, U, t, dt, F) 

    a, b, bs, c, q, Ne = butcher_arr()

    h = min(dt, step_size(V1-V2, tol, min(q), dt) )

    N = int(dt/h) + 1
    h2 = dt/N

    V1 = U
    V2 = U

    for i in range(N):
        time = t + i*dt/int(N)
        V1 = V2
        V2 = rk_scheme(1, V1, time, h2, F)

    U2 = V2

    return U2
        
        



