import numpy as np

def calculate_eigenvectors_tao(matrix):
    """
    Calculate eigenvectors using Terence Tao's method.

    Args:
    matrix (numpy.ndarray): A square symmetric matrix.

    Returns:
    numpy.ndarray: Calculated eigenvectors as columns.
    """
    N = matrix.shape[0]

    # Ensure the matrix is symmetric
    if not np.allclose(matrix, matrix.T):
        raise ValueError("Input matrix must be symmetric")

    # Calculate eigenvalues of the original matrix
    eigenvalues = np.linalg.eigvals(matrix)

    # Calculate all principal submatrices and their eigenvalues
    principal_submatrices = []
    principal_submatrices_eigenvalues = np.empty([N - 1, N])

    for j in range(N):
        principal_submatrix = np.delete(np.delete(matrix, j, axis=0), j, axis=1)
        principal_submatrices.append(principal_submatrix)
        principal_submatrices_eigenvalues[:, j] = np.linalg.eigvals(principal_submatrix)

    # Calculate eigenvectors
    eigenvectors = np.zeros((N, N))

    for i in range(N):
        for j in range(N):
            # Calculate numerator (product of eigenvalue differences)
            numerator = np.prod([eigenvalues[i] - eigenvalues[k] for k in range(N) if k != i])

            # Calculate denominator (product of eigenvalue differences with submatrix)
            denominator = np.prod([eigenvalues[i] - principal_submatrices_eigenvalues[k, j] for k in range(N - 1)])

            # Calculate squared magnitude of eigenvector component
            v_ij_squared = numerator / denominator

            # Set eigenvector component, preserving sign
            eigenvectors[j, i] = np.sqrt(np.abs(v_ij_squared)) * np.sign(v_ij_squared)

        # Normalize the eigenvector
        eigenvectors[:, i] = eigenvectors[:, i] / np.linalg.norm(eigenvectors[:, i])

    return eigenvectors

if __name__ == "__main__":
    N = 2
    # Create a random symmetric matrix
    A = np.random.rand(N, N)
    A = (A + A.T) / 2

    # Calculate eigenvectors using Tao's method
    eigenvectors_tao = calculate_eigenvectors_tao(A)

    print("Eigenvectors calculated using Tao's method:")
    print(eigenvectors_tao)

    # Compare with NumPy's eigenvector calculation
    _, eigenvectors_numpy = np.linalg.eig(A)

    print("\nEigenvectors calculated by NumPy:")
    sorted_indices = np.argsort(np.abs(eigenvectors_numpy[:, 0]))
    sorted_eigenvectors_numpy = eigenvectors_numpy[:, sorted_indices]
    print(np.abs(sorted_eigenvectors_numpy))