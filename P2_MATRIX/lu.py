import numpy as np
import matplotlib.pyplot as plt
import time

from lower_triangular import lowerTriangular
from upper_triangular import upperTriangular
from diagonal import diagonal
from permutation import permutationMatrix
from forward_substitution import forwardSubstitution
from backward_substitution import backwardSubstitution



def solveLU(A, b, x, N):

    residuum_norm = 0
    x = np.zeros(N)

    start_time = time.time()

    L = lowerTriangular(A, N)
    U = upperTriangular(A, N)
    D = diagonal(A, N)
    P = permutationMatrix(A, N)
    L = (L+D) # with diagonal
    U = (U+D) # with diagonal

    Pb = np.dot(P, b)
    
    y = forwardSubstitution(L, Pb, N)
    x = backwardSubstitution(U, y, N)

    end_time = time.time()
    elapsed_time = end_time - start_time
    rounded_time = round(elapsed_time, 3)

    residuum_norm = np.linalg.norm(np.dot(A, x) - b)
    

    print("Time cost for LU method: " + str(rounded_time) + " [s].")
    print("Residuum norm for LU method: " + str(residuum_norm))
    print()

    return rounded_time