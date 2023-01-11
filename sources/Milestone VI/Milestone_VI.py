from Cauchy_problem import Cauchy
from Temporal_schemes import RK4, Euler, Euler_Inv, Crank_Nicolson, Leap_Frog, RK_emb
from Kepler_problem import Kepler
from Lagrange import Lpoints, FL, LP_Stab

import numba
from numpy import linspace, array, zeros, size, around
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (10,6)
from random import random

@numba.jit
def main():
    
    T = 500
    dt = 0.01
    n = int(T/dt) + 1
    t = linspace(0,T,n)
    mu = 3.0039e-7

    U0LP = array([[0.8, 0.6, 0, 0],[0.8, -0.6, 0, 0],[-0.1, 0, 0, 0],[0.1, 0, 0, 0],[1.01, 0, 0, 0]])
    Np = 5      # Lagrange Points
    LPAUX = Lpoints(U0LP, Np)
    LP = zeros([5,2])
    LP[0,:] = LPAUX[3,:] #Reordeno
    LP[1,:] = LPAUX[4,:] 
    LP[2,:] = LPAUX[2,:] 
    LP[3,:] = LPAUX[0,:] 
    LP[4,:] = LPAUX[1,:] 

    labelPTot = ['L1','L2','L3','L4','L5'] #ordenados
    ShapeLP = ["<",">","d","^","v"]
    ColorLP = ["yellow","cyan","violet","sienna","lightcoral"]
    print(LP)
    for i in range(5):
        plt.plot(LP[i,0],LP[i,1],ShapeLP[i],color = ColorLP[i],label=labelPTot[i])
    plt.plot(-mu, 0, 'o', color = "g", label = 'Tierra')
    plt.plot(1-mu, 0, 'o', color = "b", label = 'Luna')
    plt.grid()
    plt.title("Puntos de Lagrange del CR3BP Tierra-Luna")
    plt.legend(loc = 'upper left',bbox_to_anchor=(1., 0.95))
    plt.savefig('MILESTONE VI media/' + 'G1 '+ str(i) +'.png')
    plt.show()


  
    U0_LP_Sel = zeros(4)
    U0_LP_SelStab = zeros(4)
    eps = 1e-4*random()
    
    for k in range(5):

        sel = k + 1 

        if sel == 1:
            labelP = 'L1'
        elif sel == 2:
            labelP = 'L2'
        elif sel == 3:
            labelP = 'L3'
        elif sel == 4:
            labelP = 'L4'
        elif sel == 5:
            labelP = 'L5'
        
        U0_LP_Sel[0:2] = LP[sel-1,:] + eps
        U0_LP_Sel[2:4] = eps

        U0_LP_SelStab[0:2] = LP[sel-1,:]
        U0_LP_SelStab[2:4] = 0

        Autoval_LP = LP_Stab(U0_LP_SelStab, mu) #estabilidad
        print(around(Autoval_LP.real,8))

        for j in range (size(1)):

            U_LP = Cauchy(FL, t, U0_LP_Sel, RK_emb) #ordenados tb

            #fig, (ax1, ax2) = plt.subplots(1, 2)
            plt.plot(U_LP[:,0], U_LP[:,1],'-',color = "k", label = 'Orbit')
            plt.plot(-mu, 0, 'o', color = "g", label = 'Tierra')
            plt.plot(1-mu, 0, 'o', color = "b", label = 'Luna')
            for i in range(5):
                plt.plot(LP[i,0],LP[i,1],ShapeLP[i],color = ColorLP[i],label=labelPTot[i])
            plt.xlim(-2,2)
            plt.ylim(-2,2)
            plt.title(f"Simulación CR3BP Tierra-Luna con esquema {RK_emb.__name__}. Órbita en {labelP}. t = {T}s, dt = {dt}. Vista completa" )    
            plt.legend(loc = 'upper left',bbox_to_anchor=(1., 0.95))
            plt.xlabel('X')
            plt.ylabel('Y')
            plt.savefig('MILESTONE VI media/' + 'G2 '+ str(k) + str(j) +'.png')
            plt.grid()
            plt.show()
                    
            plt.plot(U_LP[:,0], U_LP[:,1],'-',color = "k", label = "Orbit" )
            plt.plot(LP[sel - 1,0], LP[sel - 1,1] , ShapeLP[sel-1],color = ColorLP[sel-1], label = labelPTot[sel-1])
            plt.title(f"Simulación CR3BP Tierra-Luna con esquema {RK_emb.__name__}. Detalle de órbita en {labelP}. t = {T}s, dt = {dt}" )
            plt.xlabel('X')
            plt.ylabel('Y')
            plt.legend(loc = 'upper right',bbox_to_anchor=(1, 0.5))
            plt.savefig('MILESTONE VI media/' + 'G3 '+ str(k) + str(j) +'.png')
            plt.grid()   
            plt.xlim(LP[sel - 1,0]-0.2,LP[sel - 1,0]+0.2)
            plt.ylim(LP[sel - 1,1]-0.2,LP[sel - 1,1]+0.2)
            plt.legend(loc = 'upper left',bbox_to_anchor=(1., 0.95))
            plt.savefig('MILESTONE VI media/' + 'G4 '+ str(k) + str(j) +'.png')
            plt.show()


if __name__ == "__main__":
    main()

    input("<Hit Enter To Close>")
    plt.close('all')