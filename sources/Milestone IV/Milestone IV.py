from numpy import linspace, array, size, empty, ndarray, transpose
import numba

from Cauchy_problem import Cauchy
from Temporal_schemes import Euler, RK4, Euler_Inv, Crank_Nicolson, Leap_Frog
from Regiones_Estabilidad import stability_regions
from Oscilador import oscilation
import matplotlib.pyplot as plt

'''
Este código sirve para calcular un oscilador mediante los métodos de integración de Euler, Euler Inverso,
Cranck-Nicholson, Runge Kutta de 4º Orden y Leap-Frog

U0, T y dt son las condiciones del problema a resolver
U0 = Condiciones de contorno
T = Tiempo total
dt = Salto temporal entre un paso y el siguiente
'''

U0 = array([1, 0])
T = 20
dt = [0.1, 0.01, 0.001]

schemes = [Euler, RK4, Euler_Inv, Crank_Nicolson, Leap_Frog]
tit_scheme = ['Euler', 'Runge Kutta 4', 'Euler Inverso', 'Crank Nicolson', 'Leap Frog']

@numba.njit
def mil_IV_osc(U0, T, dt, schemes, tit_scheme):
    for j in range(size(schemes)):
        scheme = schemes[j]
        for i in range(size(dt)):
            n = int(T/dt[i])
            t = linspace(0, T, n)
            U = Cauchy(oscilation, t, U0, scheme)

            plt.plot(t, U[:,0])
        plt.title(tit_scheme[j])
        plt.xlabel("X")
        plt.ylabel("Y",rotation = 0)
        plt.legend(['dt = 0.1', 'dt = 0.01', 'dt = 0.001'])
        plt.savefig('MILESTONE IV media/' + 'Oscilator with '+ schemes[j].__name__+'.png')
        plt.show()
        plt.close()

'''
Este tramo del código se utiliza para poder calcular las regiones de estabilidad de cada uno de los métodos de integración.

a = mitad del tamaño del cuadro en el que se muestran las figuras de las regiones de estabilidad
n = refinado de las figuras mostradas
'''
a = 3
n = 100
x = linspace(-a, a, n)
y = linspace(-a, a, n)
ST = empty(len(schemes), dtype= ndarray)

@numba.njit
def milIV_stab(schemes, x, y, ST):
    for z in range(len(schemes)):
        ST[z] = stability_regions(x, y, schemes[z])
        plt.figure()
        plt.title(tit_scheme[z])
        plt.xlabel("Re")
        plt.ylabel("Im",rotation = 0)
        plt.contour(x, y, transpose(ST[z]), linspace(0, 1, 11))
        plt.draw()
        plt.savefig('MILESTONE IV media/' + 'Region '+ schemes[z].__name__+'.png')
        plt.show()
        plt.close()

if __name__ == "__mil_IV_osc__":
    mil_IV_osc(U0, T, dt, schemes, tit_scheme)
    input("<Hit Enter To Close>")
    plt.close('all')

if __name__ == "__milIV_stab__":
    milIV_stab(schemes, x, y,)
    input("<Hit Enter To Close>")
    plt.close('all')
