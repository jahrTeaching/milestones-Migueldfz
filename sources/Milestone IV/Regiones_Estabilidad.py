
from numpy import zeros, float64, abs
from Temporal_schemes import Euler, Euler_Inv, Crank_Nicolson, RK4, Leap_Frog

def stability_regions(x, y, method):
    N = len(x)
    rho =  zeros((N, N),  dtype=float64)

    for i in range(N): 
        for j in range(N):
            w = complex(x[i], y[j])
            if method == Leap_Frog:
                r = method(1, 1, 1, 0, lambda u, t: w*u)
            else:
                r = method(1, 0, 1, lambda u, t: w*u)

            rho[i, j] = abs(r)

    return rho