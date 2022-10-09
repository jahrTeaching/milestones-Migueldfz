import numpy as np
from Cauchy_problem import Cauchy
from Orbit import Kepler
from esquemas_temporales import Euler, RK4, Euler_Inv, Crank_Nicolson
import matplotlib.pyplot as plt

## Condiciones iniciales

U0 = [1, 0, 0, 1]

N = 1000
t_f = 10

t = np.linspace(0, t_f, N+1)


## Esquemas

esquema = [Euler, RK4, Euler_Inv, Crank_Nicolson]
titulo = ["Euler", "Runge-Kutta 4", "Inverse Euler", "Crank-Nicholson"]


def Simulation(t, U0, esquema, titulo): 

    i = 0

    for scheme in esquema:

        U = Cauchy(Kepler, t, U0, scheme)

        plt.plot(U[:,0],U[:,1])
        plt.title(titulo[i])
        plt.show()

        i = i+1

Simulation(t, U0, esquema, titulo)
