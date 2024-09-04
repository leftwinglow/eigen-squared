import numpy as np
from enum import Enum
# from utils import threshold
from eigen_squared.eigen_types import NumericArray, CholeskyResult
from eigen_squared.matrix_checks import is_symmetric, is_positive_definite

class CholeskyMethods(str, Enum):
    cholesky_crout = "CC"
    cholesky_banachiewicz = "CB"

class CholeskyDecomposition:
    @staticmethod
    def decompose(A: NumericArray, method: CholeskyMethods = CholeskyMethods.cholesky_banachiewicz) -> CholeskyResult:
        if not is_symmetric(A):
            raise ValueError("Matrix must be symmetric")
        if not is_positive_definite(A):
            raise ValueError("Matrix must be positive definite")

        match method:
            case CholeskyMethods.cholesky_crout:
                L = CholeskyDecomposition._Cholesky_Crout(A)
            case CholeskyMethods.cholesky_banachiewicz:
                L = CholeskyDecomposition._Cholesky_Banachieqicz(A)
            case _:
                raise ValueError(f"Invalid Cholesky method. Choose from: {', '.join([m.value for m in CholeskyMethods])}")

        return CholeskyResult(L, L.T)

    @staticmethod
    def _Cholesky_Banachieqicz(A: NumericArray) -> np.ndarray:
        n = A.shape[0]
        L = np.zeros_like(A)

        for i in range(n):
            for j in range(i+1):
                if i == j:
                    L[i,i] = np.sqrt(A[i,i] - np.sum(L[i,:i]**2))
                else:
                    L[i,j] = A[i,j] - np.sum(L[i,:j] * L[j,:j])

            # Divide the non-diagonal elements by L[j,j]
            if i > 0:
                L[i,0:i] /= np.diag(L)[:i]

        return L

    @staticmethod
    def _Cholesky_Crout(A: NumericArray) -> np.ndarray:
        n = A.shape[0]
        L = np.zeros_like(A)

        for j in range(n):
            for i in range(j, n):  # Diagonal and downwards
                if i == j:  # Diagonals, sqrt of difference btwn A and sum of squares of previous elements up to diagonal
                    sum_k = np.sum(L[i,:j]**2)
                    L[i, j] = np.sqrt(A[i, j] - sum_k)
                else:  # Off-diagonals, difference between A and sum of products of previous elements from row and col up to diagonal, divided by diagonal of current col
                    sum_k = np.sum(L[i,:j] * L[j,:j])
                    L[i, j] = (A[i, j] - sum_k) / L[j, j]

        return L