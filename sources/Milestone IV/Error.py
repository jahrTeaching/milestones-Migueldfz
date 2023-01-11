from scipy.stats import linregress
from numpy import linspace, log10, zeros, size
from numpy.linalg import norm
import matplotlib.pyplot as plt

from Cauchy_problem import Cauchy





def Richardson_error(F, Scheme, t, T, n, U0, q):

    t1 = t
    t2 = linspace(0,T,2*n)

    U1 = Cauchy(F, t1, U0, Scheme)
    U2 = Cauchy(F, t2, U0, Scheme)

    E = zeros((n, size(U0)))

    for i in range(n):
        E[i,:] = (U1[i,:] - U2[2*i,:])/(1 - 1/(2**q))

    return E



def convergence(F, scheme, t, T, n, U0, q):

    t1 = t

    m = 10
    log_E = zeros(m)
    log_N = zeros(m)
    Error = zeros(m)

    for i in range(0, m):

        t1 = linspace(0, T, n)
        t2 = linspace(0, T, 2*n)
        U1 = Cauchy(F, t1, U0, scheme)
        U2 = Cauchy(F, t2, U0, scheme)

        Error[i] = norm(U1[n-1, :] - U2[2*n-1, :])
        log_E[i] = log10(Error[i])
        log_N[i] = log10(n)

        U1 = U2
        n = 2*n


    q = linregress(log_N, log_E)

    return log_N, log_E







