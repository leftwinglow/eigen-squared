import numpy as np
from eigen_squared.eigen_types import NumericArray, EigenpairsResult
from eigen_squared.matrix_ops import calculate_rayleigh_quotient
from eigen_squared.shifts import ShiftTypes
from eigen_squared.matrix_checks import is_invertible
from tqdm import tqdm
from eigen_squared.eigenpairs.eigenvector_guesses import EigenvectorGuesses, EigenvectorGuessMethods

class RayleighEigenSolver:
    def __init__(self, A: NumericArray, max_iter: int = 5e4, tolerance: float = 1e-9, shift: ShiftTypes = ShiftTypes.wilkinson, initial_guess_method: EigenvectorGuessMethods = EigenvectorGuessMethods.last_column) -> None:
        if not is_invertible(A):
            raise ValueError("Matrix A must be invertible to calculate eigenpairs.")

        self.A = A
        self.shift = shift
        self.max_iter = int(max_iter)
        self.tolerance = float(tolerance)
        self.initial_guess_method = initial_guess_method

    def eigenpairs(self) -> EigenpairsResult:
        A = self.A.copy()
        n = self.A.shape[0]
        eigenvalues = np.zeros(n)
        eigenvectors = np.zeros((n, n))

        for self.i in tqdm(range(n)):
            v0 = EigenvectorGuesses.guess_eigenvector(A, self.initial_guess_method)
            eigenvalue, eigenvector = self._rayleigh_quotient_iterate(A, v0)
            A = A - eigenvalue * np.outer(eigenvector, eigenvector)  # Deflate the matrix

            eigenvalues[self.i] = eigenvalue
            eigenvectors[:, self.i] = eigenvector

        return EigenpairsResult(eigenvalues, eigenvectors)

    def _rayleigh_quotient_iterate(self, A: NumericArray, v: np.ndarray) -> EigenpairsResult:
        for k in range(self.max_iter):
            rq = calculate_rayleigh_quotient(A, v)
            rq_shifted = A - rq * np.eye(A.shape[0])
            try:
                y = np.linalg.solve(rq_shifted, v)  # TODO: Write a solver
            except np.linalg.LinAlgError:
                print(f"Matrix became non-singular after deflation step {self.i}. This may affect accuracy.")
                break
            new_v = y / np.linalg.norm(y)

            if np.linalg.norm(new_v - v) < self.tolerance:
                print(f"Converged after {k + 1} iterations.")
                break

            v = new_v

        else:
            print(f"Did not converge after {self.max_iter} iterations.")

        return EigenpairsResult(rq, v)

if __name__ == "__main__":
    N = 3
    A = np.random.rand(N, N)
    A = (A + A.T) / 2  # Avoid imaginary eigenvalues

    rq_algorithm = RayleighEigenSolver(A)
    my_eigenvectors = rq_algorithm.eigenpairs()
    # Compute eigenpairs using numpy for comparison
    true_eigenvalues, true_eigenvectors = np.linalg.eig(A)

    print(f"My eigenvalues: {my_eigenvectors.eigenvectors}")
    print(f"True eigenvalues: {true_eigenvectors}")

