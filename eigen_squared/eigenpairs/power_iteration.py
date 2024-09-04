import numpy as np
from eigen_squared.eigen_types import NumericArray, EigenpairsResult
from eigen_squared.matrix_ops import calculate_rayleigh_quotient
from eigen_squared.shifts import MatrixShifts, ShiftTypes
from tqdm import tqdm
from enum import Enum
from typing import Callable

class Power_Methods(str, Enum):
    power_iteration = "PI"
    inverse_power_iteration = "IPI"

class PowerEigenSolver:
    def __init__(self, A: NumericArray, max_iter: float = 5e4, tolerance: float = 1e-9, method: Power_Methods = Power_Methods.inverse_power_iteration, shift: ShiftTypes = ShiftTypes.wilkinson) -> None:
        self.A = A
        self.method = method
        self.shift = shift
        self.max_iter = int(max_iter)
        self.tolerance = float(tolerance)

    def eigenpairs(self) -> EigenpairsResult:
        """
        Compute the eigenpairs of a given matrix using the power iteration method with deflation.

        Parameters:
        A (NumericArray): The input matrix.

        Returns:
        EigenpairsResult: The eigenvalues and eigenvectors of the matrix.
        """
        A = self.A.copy()
        n = self.A.shape[0]
        eigenvalues = np.zeros(n)
        eigenvectors = np.zeros((n, n))

        match self.method:
            case Power_Methods.power_iteration:
                power_method: Callable = PowerEigenSolver._power_iterate
            case Power_Methods.inverse_power_iteration:
                power_method: Callable = PowerEigenSolver._inverse_power_iterate

        for i in tqdm(range(n)):
            eigenvalue, eigenvector = power_method(self, A)

            A = A - eigenvalue * np.outer(eigenvector, eigenvector)  # Deflate the matrix

            eigenvalues[i] = eigenvalue
            eigenvectors[:, i] = eigenvector

        return EigenpairsResult(eigenvalues, eigenvectors)

    def _power_iterate(self, A: NumericArray) -> EigenpairsResult:
        b_k = np.random.rand(A.shape[0])
        b_k[b_k == 0] = np.finfo(float).eps  # Ensure no 0 component. Non-0 components needed in direction of eigenvector
        b_k = b_k / np.linalg.norm(b_k)

        for k in range(self.max_iter):
            if k > 0 and self.shift is not None:
                shift = MatrixShifts.shift(A, self.shift)
                A -= shift * np.eye(N)
            b_k1 = np.dot(A, b_k)
            bk1_norm = np.linalg.norm(b_k1)
            b_k1 = b_k1 / bk1_norm

            if np.linalg.norm(b_k1 - b_k) < self.tolerance: # check the difference between the normalized vectors
                print(f"Converged after {k + 1} iterations.")
                break

            b_k = b_k1

        else:
            print(f"Did not converge after {self.max_iter} iterations.")

        eigenvalue = calculate_rayleigh_quotient(A, b_k)
        eigenvector = b_k

        return EigenpairsResult(eigenvalue, eigenvector)

    def _inverse_power_iterate(self, A: NumericArray) -> EigenpairsResult:
        A_inv = np.linalg.inv(A)
        b_k = np.random.rand(A_inv.shape[0])
        b_k[b_k == 0] = np.finfo(float).eps  # Ensure no 0 component. Non-0 components needed in direction of eigenvector
        b_k = b_k / np.linalg.norm(b_k)

        for k in range(self.max_iter):
            if k > 0 and self.shift:
                shift = MatrixShifts.shift(A, self.shift)
                A -= shift * np.eye(N)
            b_k1 = np.dot(A_inv, b_k)
            bk1_norm = np.linalg.norm(b_k1)
            b_k1 = b_k1 / bk1_norm

            if np.linalg.norm(b_k1 - b_k) < self.tolerance: # check the difference between the normalized vectors
                print(f"Converged after {k + 1} iterations.")
                break

            b_k = b_k1

        else:
            print(f"Did not converge after {self.max_iter} iterations.")

        eigenvalue = calculate_rayleigh_quotient(A_inv, b_k)
        eigenvector = b_k

        return EigenpairsResult(eigenvalue, eigenvector)

if __name__ == "__main__":
    N = 3
    A = np.random.rand(N, N)
    A = (A + A.T) / 2  # Avoid imaginary eigenvalues

    power_iteration_algo = PowerEigenSolver(A)
    computed_eigenpairs = power_iteration_algo.eigenpairs()
    # Compute eigenpairs using numpy for comparison
    true_eigenvalues, true_eigenvectors = np.linalg.eig(A)
    # Sort true eigenpairs by eigenvalues
    idx = np.argsort(true_eigenvalues)
    sorted_true_eigenvalues = true_eigenvalues[idx]
    sorted_true_eigenvectors = true_eigenvectors[:, idx]

    # Compare the dot products of corresponding eigenvectors
    print(f"Computed eigenvalues: {computed_eigenpairs.eigenvectors}")
    print(f"True eigenvalues: {true_eigenvectors}")