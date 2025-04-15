import numpy as np

def upperTriangular(A, N):

    result = np.zeros_like(A)

    for i in range(N - 1):
        for j in range(i + 1, N):
            result[i, j] = A[i, j]
    
    return result