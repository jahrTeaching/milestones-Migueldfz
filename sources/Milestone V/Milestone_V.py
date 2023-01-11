import matplotlib.pyplot as plt
from numpy import linspace, zeros, size
import numba

from Cauchy_problem import Cauchy
from Temporal_schemes import Euler, RK4, Euler_Inv, Crank_Nicolson, Leap_Frog
from Nbodies import init_state_nbody, Nbody

'''
Este código tiene como utilidad poder calcular el problema para Nb cuerpos con sus Nc (3) coordenadas correspondientes

Nb = Número de cuerpos
Nc = Número de coordenadas

U0, T y dt son las condiciones del problema a resolver
U0 = Condiciones de contorno
T = Tiempo total
dt = Salto temporal entre un paso y el siguiente
'''


Nb = 4                      # Number of bodies
Nc = 3                      # Number of coordinates

@numba.njit
def main():
    
    T = 40                  # Time
    dt = 0.01               # Time increase
    n = int(T/dt) + 1
    t = linspace(0,T,n)

    schemes = [Euler, RK4, Euler_Inv, Crank_Nicolson, Leap_Frog]
    tit_scheme = ['Euler', 'Runge Kutta 4', 'Euler Inverso', 'Crank Nicolson', 'Leap Frog']

    U0 = init_state_nbody(Nb,Nc)        # Initial conditions
    U = zeros([len(U0), n])             # Positions vector

    for i in range(size(schemes)):
        U = Cauchy(Nbody,t,U0,schemes[i])
        fig, twoD = plt.subplots(figsize=(5,5))
        
        
        # 2D Representation

        twoD.plot(U[:,0], U[:,2], "b")
        twoD.plot(U[:,6], U[:,8], "r")
        if Nb == 3:
            twoD.plot(U[:,12],U[:,14],"k")
        elif Nb == 4:
            twoD.plot(U[:,12],U[:,14],"k")
            twoD.plot(U[:,18],U[:,20],"purple")
        twoD.set_xlabel("X")
        twoD.set_ylabel("Y",rotation = 0)
        twoD.set_title("Proyección en el plano XY de " + str(Nb) + " cuerpos")
        plt.savefig('MILESTONE V media/' + 'Nbodies 2D with '+ schemes[i].__name__+'.png')
        twoD.grid()

        # 3D representation 

        fig = plt.figure()
        ax1 = fig.add_subplot(111,projection='3d')
        ax1.plot_wireframe(U[:,0].reshape((-1, 1)), U[:,2].reshape((-1, 1)), U[:,4].reshape((-1, 1)), color= "red", label = 'Primer cuerpo')
        ax1.plot_wireframe(U[:,6].reshape((-1, 1)), U[:,8].reshape((-1, 1)), U[:,10].reshape((-1, 1)), color= "blue", label = 'Segundo cuerpo')
        if Nb == 3:
            ax1.plot_wireframe(U[:,12].reshape((-1, 1)), U[:,14].reshape((-1, 1)), U[:,16].reshape((-1, 1)), color= "black", label = 'Tercer cuerpo')
        elif Nb == 4:
            ax1.plot_wireframe(U[:,12].reshape((-1, 1)), U[:,14].reshape((-1, 1)), U[:,16].reshape((-1, 1)), color= "black", label = 'Tercer cuerpo')
            ax1.plot_wireframe(U[:,18].reshape((-1, 1)), U[:,20].reshape((-1, 1)), U[:,22].reshape((-1, 1)), color= "purple", label = 'Cuarto cuerpo')
        
        plt.title(str(Nb) + ' cuerpos con metodo ' + tit_scheme[i] + ' y dt = ' + str(dt))
        plt.xlabel("X")
        plt.ylabel("Y",rotation = 0)
        plt.grid()
        plt.legend(loc = 'best')
        plt.savefig('MILESTONE V media/' + 'Nbodies 3D with '+ schemes[i].__name__+'.png')
        plt.show()


# RUN MAIN

if __name__ == "__main__":
    main()
    input("<Hit Enter To Close>")
    plt.close('all')














