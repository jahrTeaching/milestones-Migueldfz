import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as sp


"Dimensión de los vectores / Número de pasos"
N = 7000 
"Incremento de tiempo"
dt = 0.001 

"""
Euler
"""
x = np.array( np.zeros(N) )
y = np.array( np.zeros(N) )

"Problema Cauchy condiciones iniciales"
U = ( [ 1, 0, 0, 1] )
x[0] = U[0]
y[0] = U[1]

for i in range( 1, N):

    F = np.array( [ U[2], U[3], -U[0]/(U[0]**2 + U[1]**2)**(3/2), -U[1]/(U[0]**2 + U[1]**2)**(3/2)] )

    U = U + dt * F

    x[i] = U[0]
    y[i] = U[1]

plt.figure(1)
plt.plot( x, y)
plt.show()


"""
Runge Kutta 4º orden
"""


k1 = np.zeros(4)
k2 = np.zeros(4)
k3 = np.zeros(4)
k4 = np.zeros(4)

k1_2 = np.zeros(4)
k2_3 = np.zeros(4)
k3_4 = np.zeros(4)

x_rk = np.zeros(N)
y_rk = np.zeros(N)

"Problema Cauchy condiciones iniciales"
U_rk = ( [ 1, 0, 0, 1] )
x_rk[0] = U_rk[0]
y_rk[0] = U_rk[1]


for i in range( 1, N):

    k1 = np.array( [ U_rk[2], U_rk[3], -U_rk[0]/(U_rk[0]**2 + U_rk[1]**2)**(3/2), -U_rk[1]/(U_rk[0]**2 + U_rk[1]**2)**(3/2) ] )
    k1_2 = U_rk + dt*k1/2
    k2 = np.array( [ k1_2[2], k1_2[3], -k1_2[0]/(k1_2[0]**2 + k1_2[1]**2)**(3/2), -k1_2[1]/(k1_2[0]**2 + k1_2[1]**2)**(3/2) ] )
    k2_3 = U_rk + dt*k2/2
    k3 = np.array( [ k2_3[2], k2_3[3], -k2_3[0]/(k2_3[0]**2 + k2_3[1]**2)**(3/2), -k2_3[1]/(k2_3[0]**2 + k2_3[1]**2)**(3/2) ] )
    k3_4 = U_rk + dt*k3/2
    k4 = np.array( [ k3_4[2], k3_4[3], -k3_4[0]/(k3_4[0]**2 + k3_4[1]**2)**(3/2), -k3_4[1]/(k3_4[0]**2 + k3_4[1]**2)**(3/2) ] )

    U_rk = U_rk + dt * ( k1 + 2 * k2 + 2 * k3 + k4 ) / 6
    x_rk[i] = U_rk[0]
    y_rk[i] = U_rk[1]

plt.figure(2)
plt.plot( x_rk, y_rk)
plt.show()


"""
Crank Nicolson
"""

x_cn = np.zeros(N)
y_cn = np.zeros(N)

"Problema Cauchy condiciones iniciales"
U_cn = ( [ 1, 0, 0, 1] )
x_cn[0] = U_cn[0]
y_cn[0] = U_cn[1]

def ecs(vars):

    x,y,v,r = vars
    
    return (x - U_cn[0] - dt/2*U_cn[2] - dt/2*v,
            y - U_cn[1] - dt/2*U_cn[3] - dt/2*r,
            v - U_cn[2] - (dt/2)*(-U_cn[0]/(U_cn[0]**2 + U_cn[1]**2)**(3/2)) - (dt/2)*(-x/(x**2 + y**2)**(3/2)),
            r - U_cn[3] - (dt/2)*(-U_cn[1]/(U_cn[0]**2 + U_cn[1]**2)**(3/2)) - (dt/2)*(-y/(x**2 + y**2)**(3/2)))

for i in range( 1, N):

    x,y,v,r = sp.fsolve(ecs, (U_cn[0], U_cn[1], U_cn[2], U_cn[3]))

    U_cn = np.array([ x, y, v, r])

    x_cn[i] = x_cn[i] + U_cn[0]
    y_cn[i] = y_cn[i] + U_cn[1]


plt.figure(3)
plt.plot( x_cn, y_cn)
plt.show()
