from numpy import linspace, array, zeros
from Cauchy_problem import Cauchy
from Kepler_problem import Kepler
from Temporal_schemes import Euler, RK4, Euler_Inv, Crank_Nicolson
from Error import Richardson_error, convergence
import matplotlib.pyplot as plt


U0 = array([1, 0, 0, 1])
T = 20
dt = 0.001
n = int(T/dt)
t = linspace(0, T, n)

schemes = [Euler, RK4, Euler_Inv, Crank_Nicolson]
q_schemes = [1, 4, 1, 2]
tit_scheme = ['Euler', 'Runge Kutta 4', 'Euler Inverso', 'Crank Nicolson']


i = 0

for scheme in schemes:
    q = q_schemes[i]

    Error = Richardson_error(Kepler, scheme, t, T, n, U0, q)

    plt.plot(t, Error[:,0])
    plt.plot(t, Error[:,1])
    plt.title(tit_scheme[i])
    plt.xlabel('t [s]')
    plt.ylabel('Error')
    plt.show()

    i = i + 1

s = 0
for scheme in schemes:
    q = q_schemes[s]

    log_N, log_E = convergence(Kepler, scheme, t, T, n, U0, q)

    plt.plot(log_N, log_E)
    plt.title(tit_scheme[s])
    plt.xlabel('log(N)')
    plt.ylabel('log(U2-U1)')
    pendiente = linregress(log_N, log_E)
    plt.plot(log_N, pendiente.intercept + pendiente.slope*log_N)
    plt.show()

    s = s + 1
