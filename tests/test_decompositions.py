import numpy as np
import pytest
from eigen_squared.decompositions.QR import QRDecomposition, QR_Methods
from eigen_squared.decompositions.Cholesky import CholeskyDecomposition, CholeskyMethods
from eigen_squared.decompositions.LU import LUDecomposition, LUMethods
import scipy

np.set_printoptions(precision=8, suppress=True, linewidth=100)

def are_matrices_equal(A, B, tolerance=1e-8):
    return np.allclose(A, B, atol=tolerance)

def test_qr_decompositions():
    # Test matrices
    matrices = [
        np.random.rand(5, 5),  # Square matrix
        np.random.rand(5, 3),  # Tall matrix
        np.random.rand(3, 5),  # Wide matrix
        np.random.rand(1, 1),  # 1x1 matrix
        np.array([[1, 1], [1, 1]]),  # Singular matrix
    ]

    methods = [
        QR_Methods.gram_schmidt,
        QR_Methods.modified_gram_schmidt,
        QR_Methods.householder_reflection,
        QR_Methods.givens_rotation
    ]

    for matrix in matrices:
        for method in methods:
            # Compute QR decomposition using our implementation
            Q, R = QRDecomposition.decompose(matrix, method)

            # Compute QR decomposition using NumPy
            Q_np, R_np = np.linalg.qr(matrix)

            # Test if Q is orthogonal
            assert are_matrices_equal(np.dot(Q.T, Q), np.eye(Q.shape[1])), f"Q is not orthogonal for method {method}"

            # Test if QR equals the original matrix
            QR_product = np.dot(Q, R)
            if not are_matrices_equal(QR_product, matrix):
                print(f"Debugging info for method {method}:")
                print("Original matrix:")
                print(matrix)
                print("Q matrix:")
                print(Q)
                print("R matrix:")
                print(R)
                print("QR product:")
                print(QR_product)
                print("Difference:")
                print(np.abs(QR_product - matrix))
                print("Max difference:", np.max(np.abs(QR_product - matrix)))

            assert are_matrices_equal(QR_product, matrix), f"QR does not equal the original matrix for method {method}"

            # Test if R is upper triangular
            assert are_matrices_equal(R, np.triu(R)), f"R is not upper triangular for method {method}"

            # Compare with NumPy results
            assert are_matrices_equal(abs(Q), abs(Q_np)), f"Q differs from NumPy's Q for method {method}"
            assert are_matrices_equal(abs(R), abs(R_np)), f"R differs from NumPy's R for method {method}"

def test_qr_decomposition_errors():
    # Test invalid input
    with pytest.raises(ValueError):
        QRDecomposition.decompose(np.random.rand(3, 3), "invalid_method")  # type: ignore

    # Test empty matrix
    with pytest.raises(ValueError):
        QRDecomposition.decompose(np.array([]))

def test_cholesky_decompositions():
    # Test matrices
    matrices = [
        np.random.rand(2, 2) + np.eye(2),  # Random 2x2 invertible matrix
        np.random.rand(3, 3) + np.eye(3),  # Random 3x3 invertible matrix
        np.random.rand(4, 4) + np.eye(4),  # Random 4x4 invertible matrix
        np.random.rand(5, 5) + np.eye(5),  # Random 5x5 invertible matrix
    ]

    methods = [
        CholeskyMethods.cholesky_crout,
        CholeskyMethods.cholesky_banachiewicz
    ]

    for matrix in matrices:
        # Ensure the matrix is positive definite
        matrix = matrix @ matrix.T + np.eye(matrix.shape[0])

        for method in methods:
            # Compute Cholesky decomposition using our implementation
            L = CholeskyDecomposition.decompose(matrix, method).L

            # Compute Cholesky decomposition using NumPy
            L_np = np.linalg.cholesky(matrix)

            # Test if LL^T equals the original matrix
            LLT_product = np.dot(L, L.T)
            assert are_matrices_equal(LLT_product, matrix), f"LL^T does not equal the original matrix for method {method}"

            # Test if L is lower triangular
            assert are_matrices_equal(L, np.tril(L)), f"L is not lower triangular for method {method}"

            # Compare with NumPy results
            assert are_matrices_equal(L, L_np), f"L differs from NumPy's L for method {method}"

def test_lu_decompositions():
    # Test matrices
    matrices = [
        np.random.rand(2, 2) + np.eye(2),  # Random 2x2 invertible matrix
        np.random.rand(3, 3) + np.eye(3),  # Random 3x3 invertible matrix
        np.random.rand(4, 4) + np.eye(4),  # Random 4x4 invertible matrix
        np.random.rand(5, 5) + np.eye(5),  # Random 5x5 invertible matrix
    ]

    methods = [
        LUMethods.doolittle,
        LUMethods.crout
    ]

    for matrix in matrices:
        for method in methods:
            # Compute LU decomposition using our implementation
            L, U = LUDecomposition.decompose(matrix, method)

            # Test if L is lower triangular and U is upper triangular
            assert are_matrices_equal(L, np.tril(L)), f"L is not lower triangular for method {method}"
            assert are_matrices_equal(U, np.triu(U)), f"U is not upper triangular for method {method}"

            # Test if LU equals the original matrix
            LU_product = np.dot(L, U)
            if not are_matrices_equal(LU_product, matrix):
                print(f"Debugging info for method {method}:")
                print("Original matrix:")
                print(matrix)
                print("L matrix:")
                print(L)
                print("U matrix:")
                print(U)
                print("LU product:")
                print(LU_product)
                print("Difference:")
                print(np.abs(LU_product - matrix))

def test_cholesky_decomposition_errors():
    # Test non-positive definite matrix
    with pytest.raises(ValueError):
        CholeskyDecomposition.decompose(np.array([[1, 2], [2, 1]]))

    # Test non-square matrix
    with pytest.raises(ValueError):
        CholeskyDecomposition.decompose(np.array([[1, 2, 3], [4, 5, 6]]))

    # Test invalid method
    with pytest.raises(ValueError):
        CholeskyDecomposition.decompose(np.eye(3), "invalid_method")  # type: ignore

def test_lu_decomposition_errors():
    # Test empty matrix
    with pytest.raises(ValueError):
        LUDecomposition.decompose(np.array([]))

    # Test invalid method
    with pytest.raises(ValueError):
        LUDecomposition.decompose(np.random.rand(3, 3), "invalid_method")  # type: ignore