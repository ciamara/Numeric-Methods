import numpy as np

def permutationMatrix(A, N):

    result = np.eye(N)

    for i in range(N):
        max_row = np.argmax(np.abs(A[i:, i])) + i
        if max_row != i:
            result[[i, max_row]] = result[[max_row, i]]
            A[[i, max_row]] = A[[max_row, i]]
    

    return result