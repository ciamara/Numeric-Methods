import numpy as np
from math import sin
import matplotlib.pyplot as plt

from jacobi import solveJacobi
from gauss_seidel import solveGaussSeidel
from lu import solveLU

from create_matrix import Matrix

def main():

    # ZADANIE A i B -----------------------------------------------

    print("ZADANIE B")
    print()

    # 197584
    N = 1200 + 10*8 + 4
    a1 = 5 + 5
    a2 = -1
    a3 = -1

    
    A = Matrix(N, a1, a2, a3)

    b = [sin(n * (7 + 1)) for n in range(N)]

    x = np.ones((N, 1))

    solveJacobi(A, b, x, N, "B")

    solveGaussSeidel(A, b, x, N, "B")

    # ZADANIE C -------------------------------------------------------------

    print("ZADANIE C")
    print()

    a1 = 3
    a2 = -1
    a3 = -1

    A = Matrix(N, a1, a2, a3)

    # N i b bez zmian

    solveJacobi(A, b, x, N, "C")

    solveGaussSeidel(A, b, x, N, "C")

    # ZADANIE D -------------------------------------------------------------

    print("ZADANIE D")
    print()

    # wartosci bez zmian

    A = Matrix(N, a1, a2, a3)

    solveLU(A, b, x, N)

    # ZADANIE E ------------------------------------------------------------

    print("ZADANIE E")
    print()

    # wartosci z zad A
    # 197584
    N = [100, 500, 1000, 2000, 3000]
    a1 = 5 + 5
    a2 = -1
    a3 = -1
    A = []
    b = []
    x = []
    timesJacobi = []
    timesGaussSeidel = []
    timesLU = []

    for i, n in enumerate(N):
        b.append([sin(n * (7 + 1)) for n in range(N[i])])
        x.append(np.ones((N[i], 1)))

    for i, n in enumerate(N):
        A.append(Matrix(n, a1, a2, a3))

    for i, a in enumerate(A):
        timesJacobi.append(solveJacobi(A[i], b[i], x[i], N[i], "E"))

    for i, a in enumerate(A):
        timesGaussSeidel.append(solveGaussSeidel(A[i], b[i], x[i], N[i], "E"))

    for i, a in enumerate(A):
        timesLU.append(solveLU(A[i], b[i], x[i], N[i]))

    # wykres (liniowy y)
    plt.figure(figsize=(10, 6))
    plt.plot(N, timesJacobi, marker='o', label='Jacobi')
    plt.plot(N, timesGaussSeidel, marker='s', label='Gauss-Seidel')
    plt.plot(N, timesLU, marker='^', label='LU')
    plt.title('Czas rozwiązania w zależności od liczby niewiadomych (skala liniowa)')
    plt.xlabel('Liczba niewiadomych (N)')
    plt.ylabel('Czas [s]')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('wykres_liniowy.png')
    plt.show()

    # Wykres (log y)
    plt.figure(figsize=(10, 6))
    plt.plot(N, timesJacobi, marker='o', label='Jacobi')
    plt.plot(N, timesGaussSeidel, marker='s', label='Gauss-Seidel')
    plt.plot(N, timesLU, marker='^', label='LU')
    plt.yscale('log')
    plt.title('Czas rozwiązania w zależności od liczby niewiadomych (skala logarytmiczna)')
    plt.xlabel('Liczba niewiadomych (N)')
    plt.ylabel('Czas [s] (skala logarytmiczna)')
    plt.legend()
    plt.grid(True, which='both', linestyle='--')
    plt.tight_layout()
    plt.savefig('wykres_logarytmiczny.png')
    plt.show()


if __name__ == "__main__":
    main()
