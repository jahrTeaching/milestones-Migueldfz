from numpy import zeros
from Temporal_schemes import Euler, Euler_Inv, Crank_Nicolson, RK4, Leap_Frog

def Cauchy(F , t, U0, Esquema_temp):

    N, Nv = len(t)-1, len(U0)
    U = zeros((N+1, Nv))
    U[0,:] = U0
    dt = t[1] - t[0]

    if Esquema_temp == Leap_Frog:
        U[1,:] = U[0,:] + dt*F(U[0,:], t[0])

        for n in range(1,N):
            U1 = U[n-1, :]
            U2 = U[n, :]
            U[n+1, :] = Esquema_temp(U1, U2, dt, t[n], F)

    else:
        for n in range(N):

            U[n+1,:] = Esquema_temp(U[n,:], t[n], t[n+1] - t[n], F)

    return U
