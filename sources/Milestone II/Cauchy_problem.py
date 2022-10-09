import numpy as np


def Cauchy(F , t, U0, Esquema_temp):

    N, Nv = len(t)-1, len(U0)    # Nº de intervalos y Nº de variables  

    U = np.zeros((N+1, Nv))

    U[0,:] = U0

    for n in range(N):

        U[n+1,:] = Esquema_temp(U[n,:], t[n], t[n+1] - t[n], F)

    return U
