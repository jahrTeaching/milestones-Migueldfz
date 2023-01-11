from numpy import array


def oscilation(U,t):
    return array([U[1], -U[0]])