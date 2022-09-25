import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as sp

def kepler_abstraction_euler():

    U0 = np.array( [ 1, 0, 0, 1] )
    N = 150
    dt = 0.1

    U = kepler_euler( U0, dt, N)




def kepler_euler( U0, dt, N):

    nu = len(U0)    
    U = np.zeros(nu)
    U = U0
    x = np.zeros(N)
    y = np.zeros(N)
    x[0] = U0[0]
    y[0] = U0[1]
    for i in range(N):

        F = F_euler(U,i)
        U = U_euler(U,F,dt,i)
        x[i] = U[0]
        y[i] = U[1]
        
        
    plt.plot( x, y)
    plt.show()


def U_euler(U,F,dt,i):

    return U + dt * F

def F_euler(U,i):

    x = U[0]; y = U[1]; dx_dt = U[2]; dy_dt = U[3]
    d = ( x**2 + y**2 )**1.5


    return np.array([ float(dx_dt), float(dy_dt), float(-x/d), float(-y/d)])
