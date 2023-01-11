from numpy import concatenate, zeros, reshape, linalg, array, sqrt
#from Milestone_V import Nb, Nc
Nb = 4
Nc = 3

def init_state_nbody(Nb, Nc):
    U0 = zeros(Nb*Nc*2)

    if Nb == 2:                     # 2 bodies
        r01 = array([1, 0, 0])
        v01 = array([0, -0.5, 0])

        r02 = array([-0.5, 0, 0])
        v02 = array([0, 0.5, 0])

        r = concatenate((r01,r02),axis=0)
        v = concatenate((v01,v02),axis=0)

    elif Nb == 3:                      # 4 bodies, 

        r01 = array([5, 0, 0])/11
        v01 = array([0, 1, 0])

        r02 = array([-2.5, 4.33012701892, 0])/11
        v02 = array([-sqrt(3)/2, -0.5, 0])

        r03 = array([-2.5, -4.33012701892, 0])/11
        v03 = array([sqrt(3)/2, -0.5, 0])

        r = concatenate((r01,r02,r03),axis=0)
        v = concatenate((v01,v02,v03),axis=0)

    elif Nb == 4:                     # 4 bodies
        
        # Body 1
        r01 = array([2, 2, 1])
        v01 = array([-0.5, 0, 0])

        # Body 2
        r02 = array([-2,2,-1])
        v02 = array([0,-0.5,0])

        # Body 3
        r03 = array([-2, -2, 1])
        v03 = array([0.5, 0, 0])

        # Body 4
        r04 = array([2, -2, -1])
        v04 = array([0, 0.5, 0])

        r = concatenate((r01,r02,r03,r04),axis=0)
        v = concatenate((v01,v02,v03,v04),axis=0)

    for i in range(len(r)):     # U0 built with the position and velocities in each direction, aka. [x, x', y, y', z, z']

        U0[2*i] = r[i]
        U0[2*i + 1] = v[i]

    return U0


def Nbody(U, t):

    (Nb,Nc) = (4,3)    

    Us = reshape(U,(Nb,Nc,2))           
    F  = zeros(len(U))
    Fs = reshape(F,(Nb,Nc,2))           

    r  = reshape(Us[:,:,0],(Nb,Nc))     # N-Body locations
    v  = reshape(Us[:,:,1],(Nb,Nc))     # N-Body velocities

    drdt = reshape(Fs[:,:,0], (Nb,Nc))
    dvdt = reshape(Fs[:,:,1], (Nb,Nc))

    dvdt[:,:] = 0                 

    for i in range(Nb):

        drdt[i,:] = v[i,:]

        for j in range(Nb):
            
            if j != i:                   # Different bodies atract eachother

                d = r[j,:] - r[i,:]
                dvdt[i,:] += d[:]/(linalg.norm(d)**3)  

    return F
