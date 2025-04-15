import numpy as np
import matplotlib.pyplot as plt
import time

from lower_triangular import lowerTriangular
from upper_triangular import upperTriangular
from diagonal import diagonal


def solveJacobi(A, b, x, N, exercise):

    iteration_count = 0
    residuum_norm = []

    start_time = time.time()

    #D = diagonal(A, N)
    D = np.diag(np.diag(A))
    #L = lowerTriangular(A, N)
    L = np.tril(A)
    L = (L-D)
    #U = upperTriangular(A, N)
    U = np.triu(A)
    U = (U-D)
    

    M = -np.linalg.solve(D, L + U)
    w = np.linalg.solve(D, b)

    inorm = np.linalg.norm(np.dot(A, x) - b)
    residuum_norm = [inorm]
    
    while inorm > 1e-9 and iteration_count < 100:

        x = np.dot(M, x) + w
        inorm = np.linalg.norm(np.dot(A, x) - b)

        iteration_count += 1
        residuum_norm.append(inorm)

    end_time = time.time()
    elapsed_time = end_time - start_time
    rounded_time = round(elapsed_time, 3)
    

    print("Time cost for Jacobi method: " + str(rounded_time) + " [s].")
    print("Final residuum norm for Jacobi method: " + str(residuum_norm[-1]))
    print("Iteration count for Jacobi method: " + str(iteration_count))
    print()

    plt.plot(np.arange(iteration_count + 1), residuum_norm, linewidth=1.5, color="pink")
    plt.yscale('log')
    plt.xlabel('Iteration count')
    plt.ylabel('Residuum norm ||A*x - b||')
    if exercise == "B":
        plt.title('Convergence of Jacobi method ( zadanie B)')
    elif exercise == "C":
        plt.title('Convergence of Jacobi method ( zadanie C)')

    plt.grid(True)

    if exercise == "B":
        plt.savefig('jacobi_convergence_B.png', format='png')
    elif exercise == "C":
        plt.savefig('jacobi_convergence_C.png', format='png')
    
    if exercise != "E":
        plt.show()

    return rounded_time
