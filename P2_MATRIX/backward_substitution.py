import numpy as np

def backwardSubstitution(U, y, N):

    x = np.zeros(N)

    for i in reversed(range(N)):
        sum_Ux = sum(U[i][j] * x[j] for j in range(i + 1, N))
        x[i] = (y[i] - sum_Ux) / U[i][i]
        
    return x