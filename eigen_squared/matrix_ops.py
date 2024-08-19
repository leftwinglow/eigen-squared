import numpy as np
from eigen_squared.eigen_types import NumericArray
import eigen_squared.matrix_checks as MC

def pivot_columns(A: NumericArray) -> tuple[np.ndarray, np.ndarray]:
    """
    Perform column pivoting on the input matrix.
    Parameters:
    A (np.ndarray): The input matrix.
    Returns:
    tuple[np.ndarray, np.ndarray]: A tuple containing the pivoted matrix and the permutation matrix.
    """
    m, n = A.shape
    P = np.eye(n)
    A_pivoted = A.copy()
    # If A has more rows than columns (m > n), we only need to pivot n-1 times
    # If A n > m, we only need to pivot m-1 times as remaining columns dont affect upper triangular R
    for i in range(min(m - 1, n)):  # We don't need to pivot the last column
        # Find the column with the largest norm in the remaining submatrix
        norms = np.linalg.norm(A_pivoted[i:, i:], axis=0)
        pivot = np.argmax(norms) + i
        # Swap columns
        A_pivoted[:, [i, pivot]] = A_pivoted[:, [pivot, i]]
        P[:, [i, pivot]] = P[:, [pivot, i]]
    return (A_pivoted, P)

def calculate_rayleigh_quotient(A: NumericArray, x: NumericArray) -> float:
    """
    Calculate the Rayleigh quotient for a given matrix A and vector x.
    Parameters:
    A (NumericArray): The input matrix.
    x (NumericArray): The input vector.
    Returns:
    float: The Rayleigh quotient.
    """
    if not MC.is_square(A):
        raise ValueError("Matrix A must be square to calculate rayleigh quotient.")
    if not MC.is_vector(x):
        raise ValueError("Input x must be a vector to calculate rayleigh quotient.")
    if not A.shape[0] == x.shape[0]:
        raise ValueError("Matrix A and vector x must have the same dimensions.")
    if np.allclose(x, 0):
        raise ValueError("Vector x must not be the zero vector.")
    return np.dot(x, A @ x) / np.dot(x, x)