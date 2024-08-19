import numpy as np
from typing import Optional
from eigen_squared import matrix_checks as MC
from eigen_squared.matrix_ops import calculate_rayleigh_quotient
from eigen_squared.eigen_types import NumericArray
from enum import Enum

class ShiftTypes(str, Enum):
    simple = "simple"
    spectral = "spectral"
    wilkinson = "wilkinson"
    rayleigh_quotient = "rayleigh_quotient"

class MatrixShifts:
    def shift(A: NumericArray, shift_type: ShiftTypes, sigma: Optional[float] = None) -> float:
        match shift_type:
            case ShiftTypes.simple:
                return MatrixShifts._simple_shift(A)
            case ShiftTypes.spectral:
                if sigma is None:
                    raise ValueError("Spectral shift requires a shift value.")
                return MatrixShifts._spectral_shift(A, sigma)
            case ShiftTypes.wilkinson:
                return MatrixShifts._wilkinson_shift(A)
            case ShiftTypes.rayleigh_quotient:
                return MatrixShifts._rayleigh_quotient_shift(A, A[:, -1])
            case _:
                raise ValueError(f"Invalid shift type. Choose from: {', '.join([shift.value for shift in ShiftTypes])}")

    def _simple_shift(A: NumericArray) -> float:
        if not MC.is_square(A):
            raise ValueError("Matrix A must be square to calculate simple shift.")
        return A[-1, -1]

    def _spectral_shift(A: NumericArray, sigma: float) -> float:
        if not MC.is_square(A):
            raise ValueError("Matrix A must be square to calculate spectral shift.")
        return sigma

    def _wilkinson_shift(A: NumericArray) -> float:
        if MC.is_vector(A):
            raise ValueError("Input must be a matrix to calculate the Wilkinson shift.")
        d = (A[-2, -2] - A[-1, -1]) / 2  # Bottom left 2x2 submatrix / 2, most important for eigenvalues of A
        sign = np.sign(d) if d != 0 else 1  # Sign of d, but 1 if d is 0
        mu = A[-1, -1] - sign * A[-1, -2]**2 / (abs(d) + np.sqrt(d**2 + A[-1, -2]**2))
        return mu

    def _rayleigh_quotient_shift(A: NumericArray, x: NumericArray) -> float:
        """
        Calculates the Rayleigh quotient shift for a given matrix A and vector x.

        Parameters:
        A (NumericArray): The Hermitian matrix A.
        x (NumericArray): The initial guess of the eigenvector x.

        Returns:
        float: The Rayleigh quotient shift.

        Raises:
        ValueError: If the matrix A is not square or if the input x is not a vector.
        """
        return calculate_rayleigh_quotient(A, x)

# TODO: Move
def gershgorin_discs(A: NumericArray) -> np.ndarray:
    """
    Calculate the Gershgorin discs for a given square matrix. Every eigenvalue lies within the union of these discs.

    Parameters:
    A (NumericArray): The square matrix for which to calculate the Gershgorin discs.

    Returns:
    np.ndarray: A tuple containing the centers and radii of the Gershgorin discs.

    """
    if not MC.is_square(A):
        raise ValueError("Matrix A must be square to calculate Gershgorin discs.")

    centers = np.diag(A)
    radii = np.sum(abs(A), axis=1) - centers
    return centers, radii