import numpy as np

def lowerTriangular(A, N):

    result = np.zeros_like(A)

    for i in range(1, N):
        for j in range(i):
            result[i, j] = A[i, j]

    return result