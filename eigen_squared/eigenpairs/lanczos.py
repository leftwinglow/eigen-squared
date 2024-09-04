import numpy as np
from eigen_squared.eigen_types import NumericArray
from typing import Tuple
from eigen_squared.eigenpairs.eigenvector_guesses import EigenvectorGuessMethods, EigenvectorGuesses
from eigen_squared.matrix_checks import is_invertible
from eigen_squared.eigen_types import EigenpairsResult


class LanczosEigenSolver:
    def __init__(self, A: NumericArray, max_iter: float = 5e5, tolerance: float = 1e-9,
                 initial_guess_method: EigenvectorGuessMethods = EigenvectorGuessMethods.last_column) -> None:
        if not is_invertible(A):
            raise ValueError("Matrix A must be invertible to calculate eigenpairs.")

        self.A = A
        self.max_iter = int(max_iter)
        self.tolerance = float(tolerance)
        self.initial_guess_method = initial_guess_method

    def eigenpairs(self) -> EigenpairsResult:
        A = self.A.copy()

        v0 = EigenvectorGuesses.guess_eigenvector(A, self.initial_guess_method)
        eigenvector, eigenvalue = self._lanczos_iteration(A, v0)

        # Sort eigenpairs by eigenvalue magnitude
        # idx = np.argsort(np.abs(eigenvalues))[::-1]
        return EigenpairsResult(eigenvector, eigenvalue)

    def _lanczos_iteration(self, A: NumericArray, v0: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        n = A.shape[0]
        V = np.zeros((n, n))
        T = np.zeros((n, n))

        v0 = v0 / np.linalg.norm(v0)
        V[:, 0] = v0

        for j in range(n):
            w = A @ V[:, j]

            # Full reorthogonalization
            for i in range(j + 1):
                T[i, j] = V[:, i].T @ w
                w = w - T[i, j] * V[:, i]

            if j < n - 1:
                T[j+1, j] = np.linalg.norm(w)
                if T[j+1, j] < self.tolerance:
                    break
                V[:, j+1] = w / T[j+1, j]

        # Extract eigenpairs from T
        eigenvalues, eigenvectors = np.linalg.eigh(T[:j+1, :j+1])

        # Transform eigenvectors back to original space
        eigenvectors = V[:, :j+1] @ eigenvectors

        return eigenvalues, eigenvectors

if __name__ == "__main__":
    N = 3
    A = np.random.rand(N, N)
    A = (A + A.T) / 2  # Avoid imaginary eigenvalues

    computed_eigenpairs = LanczosEigenSolver(A).eigenpairs()
    # Compute eigenpairs using numpy for comparison
    true_eigenvalues, true_eigenvectors = np.linalg.eig(A)
    # Sort true eigenpairs by eigenvalues
    idx = np.argsort(true_eigenvalues)
    sorted_true_eigenvalues = true_eigenvalues[idx]
    sorted_true_eigenvectors = true_eigenvectors[:, idx]

    # Compare the dot products of corresponding eigenvectors
    dot_product = np.abs(np.dot(computed_eigenpairs[1], sorted_true_eigenvectors[1]))
    print(f"Absolute dot product of eigenvector: {dot_product}")
    print(f"Matrix: \n {A} \n \n My Lanczos: \n {computed_eigenpairs.eigenvectors} \n\n Numpy: \n{true_eigenvectors}")