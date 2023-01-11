from numpy import zeros, sqrt, array, linalg, size
from scipy.optimize import fsolve
import numba


def CR3BP(U): 
    mu = 3.0039e-7
    
    x = U[0]; y = U[1]
    vx = U[2]; vy = U[3]
    
    r1 = sqrt( (x+mu)**2 + y**2 )
    r2 = sqrt( (x-1+mu)**2 + y**2 )
    
    dxdt, dydt = vx, vy
    
    dvxdt = 2*vy+x-( (1-mu)*(x+mu) ) / (r1**3) - mu*(x+mu-1)/(r2**3)
    dvydt = -2*vx + y -( (1-mu) / (r1**3) + mu/(r2**3) )*y
    
    return array([ dxdt, dydt, dvxdt, dvydt])


def FL(U,t):                 # Wrapped de la función CR3BP
    return CR3BP(U)


def Lpoints(U0, Np):
    LP = zeros([Np,2])

    def F(Y):
        X = zeros(4)
        X[0:2] = Y
        X[2:4] = 0
        FX = CR3BP(X)
        return FX[2:4]
   
    for i in range(Np):
        LP[i,:] = fsolve(F, U0[i,0:2])

    return LP


def Jac(F, U):
	N = size(U)
	J= zeros([N,N])
	t = 1e-10

	for i in range(N):
		xj = zeros(N)
		xj[i] = t
		J[:,i] = (F(U + xj) - F(U - xj))/(2*t)
	return J  


def LP_Stab( U0, mu ):

    A = Jac(CR3BP, U0)
    values, vectors = linalg.eig(A)

    return values



