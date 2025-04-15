import numpy as np

def forwardSubstitution(L, b, N):

    y = np.zeros(N)

    for i in range(N):
        sum_Ly = sum(L[i][j] * y[j] for j in range(i))
        y[i] = (b[i] - sum_Ly) / L[i][i]
        
    return y