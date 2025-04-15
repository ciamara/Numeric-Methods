import numpy as np

def Matrix(N, a1, a2, a3):

    matrix = np.zeros((N, N))

    for i in range(N):
        matrix[i, i] = a1
    
    for i in range(N - 1):
        matrix[i, i + 1] = a2
        matrix[i + 1, i] = a2
    
    for i in range(N - 2):
        matrix[i, i + 2] = a3
        matrix[i + 2, i] = a3
    
    return matrix