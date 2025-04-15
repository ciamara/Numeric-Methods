import numpy as np

def diagonal(A, N):

    result = np.zeros_like(A.data)
    
    for i in range(N):
        result[i, i] = A[i, i]
    
    return result