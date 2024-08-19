import numpy as np
from enum import Enum

class EigenvectorGuessMethods(str, Enum):
    first_column = "first_column"
    last_column = "last_column"
    random_column = "random_column"
    mean_column = "mean_column"
    random_vector = "random_vector"

class EigenvectorGuesses:
    @staticmethod
    def guess_eigenvector(A: np.ndarray, guess: str) -> np.ndarray:
        match guess:
            case EigenvectorGuessMethods.first_column:
                return EigenvectorGuesses._guess_first_column(A)
            case EigenvectorGuessMethods.last_column:
                return EigenvectorGuesses._guess_last_column(A)
            case EigenvectorGuessMethods.random_column:
                return EigenvectorGuesses._guess_random_column(A)
            case EigenvectorGuessMethods.mean_column:
                return EigenvectorGuesses._guess_mean_column(A)
            case EigenvectorGuessMethods.random_vector:
                return EigenvectorGuesses._guess_random_vector(A)

    def _guess_first_column(A: np.ndarray) -> np.ndarray:
        """Guess the first eigenvector."""
        n = A[:, 0]
        return n

    def _guess_last_column(A: np.ndarray) -> np.ndarray:
        """Guess the last eigenvector."""
        n = A[:, -1]
        return n

    def _guess_random_column(A: np.ndarray) -> np.ndarray:
        """Guess a random eigenvector."""
        rand_col = np.random.randint(A.shape[1])
        n = A[:, rand_col]
        return n

    def _guess_mean_column(A: np.ndarray) -> np.ndarray:
        """Guess the mean of the eigenvectors."""
        n = np.mean(A, axis=1)
        return n

    def _guess_random_vector(A: np.ndarray) -> np.ndarray:
        """Guess a random eigenvector."""
        n = np.random.rand(A.shape[0])
        return n

if __name__ == "__main__":
    N = 3
    A = np.random.rand(N, N)
    A = (A + A.T) / 2  # Avoid imaginary eigenvalues