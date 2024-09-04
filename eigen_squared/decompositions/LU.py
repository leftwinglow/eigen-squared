import numpy as np
from enum import Enum
from eigen_squared.eigen_types import NumericArray, LUResult

class LUMethods(str, Enum):
    doolittle = "Doolittle"
    crout = "Crout"

class LUDecomposition:
    @staticmethod
    def decompose(A: NumericArray, method: LUMethods = LUMethods.doolittle) -> LUResult:
        match method:
            case LUMethods.doolittle:
                L, U = LUDecomposition._LU_Doolittle(A)
            case LUMethods.crout:
                L, U = LUDecomposition._LU_Crout(A)
            case _:
                raise ValueError(f"Invalid method. Choose from: {', '.join([method.value for method in LUMethods])}")

        return LUResult(L, U)

    @staticmethod
    def _LU_Doolittle(A: NumericArray) -> tuple[np.ndarray, np.ndarray]:
        n = A.shape[0]
        L = np.eye(n)
        U = np.zeros_like(A)

        for i in range(n):
            for j in range(i, n):
                #  U[i, j] means all elements above the diagonal in the ith row
                U[i, j] = A[i, j] - np.sum(L[i, :i] * U[:i, j])  # Sum previous cols up to col i

            for j in range(i + 1, n):
                L[j, i] = (A[j, i] - np.sum(L[j, :i] * U[:i, i])) / U[i, i]  # Sum previous rows up to row i

        assert np.allclose(A, L @ U) and np.allclose(L, np.tril(L)) and np.allclose(U, np.triu(U))

        return L, U

    @staticmethod
    def _LU_Crout(A: NumericArray) -> tuple[np.ndarray, np.ndarray]:
        n = A.shape[0]
        L = np.eye(n)
        U = np.zeros_like(A)

        for j in range(n):
            for i in range(j, n):
                # L[i, j] means all elements in the jth column of L
                L[i, j] = A[i, j] - np.sum(L[i, :j] * U[:j, j])

            for i in range(j + 1, n):
                # U[j, i] means all elements to the right of the diagonal in the jth row of U
                U[j, i] = (A[j, i] - np.sum(L[j, :j] * U[:j, i])) / L[j, j]

        U[n-1, n-1] = (A[n-1, n-1] - np.sum(L[n-1, :n-1] * U[:n-1, n-1])) / L[n-1, n-1]

        assert np.allclose(A, L @ U) and np.allclose(L, np.tril(L)) and np.allclose(U, np.triu(U))

        return L, U