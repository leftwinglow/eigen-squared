import numpy as np
from enum import Enum
from .eigen_types import NumericArray
from typing import Literal, Callable

class PosDefMethods(str, Enum):
    strict_diagonally_dominant = "SDD"
    gershgorin = "Gershgorin"

class InvertibleMethods(str, Enum):
    determinant = "det"
    rank = "rank"
    eigenvalues = "eigenvalues"

def is_vector(vector: NumericArray) -> bool:
    """Check if the input is a vector."""
    return len(vector.shape) == 1

def is_square(matrix: NumericArray) -> bool:
    """Check if the matrix is square."""
    return matrix.shape[0] == matrix.shape[1]

def is_invertible(matrix: NumericArray, method: InvertibleMethods = InvertibleMethods.rank) -> bool:
    """Check if the matrix is invertible."""
    match method:
        case InvertibleMethods.determinant:
            return np.linalg.det(matrix) != 0
        case InvertibleMethods.rank:
            return np.linalg.matrix_rank(matrix) == matrix.shape[0]
        case InvertibleMethods.eigenvalues:
            return np.all(np.linalg.eigvals(matrix) != 0)  # TODO: Use my eigenpairs function
        case _:
            raise ValueError(f"Invalid invertible method. Choose from: {', '.join([m.value for m in InvertibleMethods])}")

def is_symmetric(matrix: NumericArray) -> bool:
    """Check if the matrix is symmetric."""
    if not is_square(matrix):
        raise ValueError("Non-square matrices cannot be symmetric")
    return np.allclose(matrix, matrix.T)

def is_hermitian(matrix: NumericArray) -> bool:
    """Check if the matrix is Hermitian."""
    if not is_square(matrix):
        raise ValueError("Non-square matrices cannot be Hermitian")
    return np.allclose(matrix, np.conjugate(matrix.T))

def is_positive_definite(matrix: NumericArray, method: PosDefMethods = PosDefMethods.strict_diagonally_dominant) -> bool:
    """Check if the matrix is positive definite."""

    if not is_square(matrix):
        raise ValueError("Non-square matrix cannot be positive definite")

    n = matrix.shape[0]
    match method:
        case PosDefMethods.strict_diagonally_dominant:
            return all(is_diagonally_dominant(matrix, 'row', 'strict') and matrix[i,i] > 0 for i in range(n))
        case _:
            raise ValueError(f"Invalid positive definite method. Choose from: {', '.join([m.value for m in PosDefMethods])}")

def is_diagonally_dominant(
    matrix: NumericArray,
    direction: Literal['row', 'column'],
    strictness: Literal['strict', 'weak']
) -> bool:
    """
    Check if the matrix is diagonally dominant.

    Args:
    matrix: The input matrix
    direction: 'row' for row dominance, 'column' for column dominance
    strictness: 'strict' for strictly dominant, 'weak' for weakly dominant

    Returns:
    bool: True if the matrix is diagonally dominant, False otherwise
    """
    n = matrix.shape[0]

    def strict_compare(x: float, y: float) -> bool:
        return x > y

    def weak_compare(x: float, y: float) -> bool:
        return x >= y

    compare: Callable[[float, float], bool] = (
        strict_compare if strictness == 'strict' else weak_compare
    )

    for i in range(n):
        if direction == 'row':
            sum_non_diag = np.sum(np.abs(matrix[i, :])) - np.abs(matrix[i, i])
            if not compare(np.abs(matrix[i, i]), sum_non_diag):
                return False
        else:  # column
            sum_non_diag = np.sum(np.abs(matrix[:, i])) - np.abs(matrix[i, i])
            if not compare(np.abs(matrix[i, i]), sum_non_diag):
                return False
    return True

def is_diagonal(matrix: NumericArray) -> bool:
    """Check if the matrix is diagonal."""
    return np.allclose(matrix, np.diag(np.diagonal(matrix)))