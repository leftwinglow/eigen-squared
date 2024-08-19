import numpy as np
from eigen_squared.decompositions.QR import QRDecomposition, QR_Methods
from eigen_squared.shifts import MatrixShifts, ShiftTypes
from eigen_squared.eigen_types import NumericArray, EigenpairsResult, QRResult
from eigen_squared.norms import FrobeniusNorms as FN
from eigen_squared.matrix_ops import pivot_columns
from tqdm import tqdm

class QREigenSolver:
    def __init__(self, A: NumericArray, max_iter: int = 5e4, tolerance: float = 1e-9, method: QR_Methods = QR_Methods.householder_reflection, shift: ShiftTypes = ShiftTypes.wilkinson, pivot: bool = False):
        self.A = A
        if pivot:
            self.A, self.P = pivot_columns(self.A)
        else:
            self.P = np.eye(A.shape[1])

        self.method = method
        self.shift = shift
        self.max_iter = int(max_iter)
        self.tolerance = float(tolerance)

    def eigenpairs(self) -> EigenpairsResult:
        Ak = self.A.copy()
        N = Ak.shape[0]
        Q_accumulated = np.eye(N)
        shift = 0

        for k in tqdm(range(self.max_iter)):
            if k > 0 and self.shift is not None:
                shift = MatrixShifts.shift(Ak, self.shift)
                Ak -= shift * np.eye(N)

            Q, R = QRDecomposition.decompose(Ak, self.method)
            Ak = R @ Q + shift * np.eye(N)

            Q_accumulated = Q_accumulated @ Q

            if FN.frobenius_norm(Ak, "non_diag") < self.tolerance:
                print(f"Converged after {k + 1} iterations.")
                break
        else:
            print(f"Did not converge after {self.max_iter} iterations.")

        eigenvectors = np.transpose(self.P) @ Q_accumulated
        eigenvalues = np.diag(Ak)

        return EigenpairsResult(eigenvalues, eigenvectors)

    def decompose_wrapper(self) -> QRResult:
        return QRDecomposition.decompose(self.A, self.method)

if __name__ == "__main__":
    N = 3
    A = np.random.rand(N, N)
    A = (A + A.T) / 2  # Avoid imaginary eigenvalues

    qr_algorithm = QREigenSolver(A)
    computed_eigenpairs = qr_algorithm.eigenpairs()
    # Compute eigenpairs using numpy for comparison
    true_eigenvalues, true_eigenvectors = np.linalg.eig(A)
    # Sort true eigenpairs by eigenvalues
    idx = np.argsort(true_eigenvalues)
    sorted_true_eigenvalues = true_eigenvalues[idx]
    sorted_true_eigenvectors = true_eigenvectors[:, idx]

    # Compare the dot products of corresponding eigenvectors
    for i in range(A.shape[0]):
        dot_product = np.abs(np.dot(computed_eigenpairs.eigenvectors[:, i], sorted_true_eigenvectors[:, i]))
        print(f"Absolute dot product of eigenvector {i}: {dot_product:.4f}")

    # The dot products of both should be ~1 if they lie on the same plane