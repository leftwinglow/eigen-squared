import numpy as np
from eigen_squared.eigen_types import NumericArray, QRResult
from enum import Enum

class QR_Methods(str, Enum):
    gram_schmidt = "GS"
    modified_gram_schmidt = "MGS"
    householder_reflection = "HR"
    givens_rotation = "GR"

class QRDecomposition:
    @staticmethod
    def decompose(A: NumericArray, method: QR_Methods = QR_Methods.modified_gram_schmidt) -> QRResult:
        """
        Decomposes a matrix A into its QR factorization.

        Args:
            A (NumericArray): The matrix to be decomposed.
            method (QR_Methods, optional): The method to be used for the QR decomposition. Defaults to "MGS".
            pivot (bool, optional): Whether to perform column pivoting. Defaults to False.

        Returns:
            tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray]: The QR factorization of the matrix A.
                If pivot is True, returns Q, R, P.
                If pivot is False, returns Q, R.
        """

        match method:
            case QR_Methods.gram_schmidt:
                Q,R = QRDecomposition._gram_schmidt(A)
            case QR_Methods.modified_gram_schmidt:
                Q,R = QRDecomposition._modified_gram_schmidt(A)
            case QR_Methods.householder_reflection:
                Q,R = QRDecomposition._householder_reflection(A)
            case QR_Methods.givens_rotation:
                Q,R = QRDecomposition._givens_rotation(A)
            case _:
                raise ValueError(f"Invalid method. Choose from: {', '.join([method.value for method in QR_Methods])}")

        return QRResult(Q, R)

    @staticmethod
    def _gram_schmidt(A: NumericArray) -> QRResult:
        """
        Compute the QR decomposition of a matrix using Gram-Schmidt.

        Parameters:
        A (np.ndarray): The input matrix.

        Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing the orthogonal matrix Q and the upper triangular matrix R.
        """
        n, m = A.shape
        Q = np.empty((n, m))  # Orthonormal basis
        R = np.zeros((m, m))  # Will be upper triangular
        u = np.empty((n, n))  # Orthogonal vectors (normalised e vectors)

        # First entries in Q, U
        u[:, 0] = A[:, 0]
        Q[:, 0] = u[:, 0] / np.linalg.norm(u[:, 0])

        for col in range(1, n):
            u[:, col] = A[:, col]

            #  Compute the u vectors
            for i in range(col):
                # Projecting A[:, col] onto Q[:, i], (a2 \cdot e1)*e1
                projection = np.dot(Q[:, i], A[:, col]) * Q[:, i]

                # The part of the equation in asterixes *-a2 - (a2 \cdot e1)*e1*
                # This works because we set u[:, col] = A[:, col] at the beginning, then subtract the projection, so we get the orthogonal vector in u[:, col]
                u[:, col] -= projection

            # Compute the e vectors; i.e., normalize the u vectors
            Q[:, col] = u[:, col] / np.linalg.norm(u[:, col])

        # R[i, j], how much of the j-th column of A is in the i-th column of Q
        for i in range(n):  # for eah row in Q
            for j in range(i, m):  # for each column in A
                R[i, j] = np.dot(Q[:, i], A[:, j])  # R is the dot product of the ith column of Q and the jth column of A

        return QRResult(Q, R)

    @staticmethod
    def _modified_gram_schmidt(A: NumericArray) -> QRResult:
        """
        Compute the QR decomposition of a matrix using modified Gram-Schmidt.

        Parameters:
        A (np.ndarray): The input matrix.

        Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing the orthogonal matrix Q and the upper triangular matrix R.
        """
        n, m = A.shape
        Q = np.empty((n, n))
        R = np.zeros((n, m))

        #  On the k-th iteration, we take the orthogonalized A[:, k] and normalize it to get the k-th column of Q
        #  Hence, each new column of Q is orthogonal to the previous columns
        #  Basically the k-th col of A is orthogonalised to the previous k-1 columns of Q
        for k in range(m):
            norm = np.linalg.norm(A[:, k])  # Calculate the norm of the k-th column of A
            if norm == 0:
                # Handle the zero norm case
                R[k, k] = 0
                Q[:, k] = 0  # Or some other appropriate value
            else:
                R[k, k] = norm  # Use the norm of the k-th column of A as the diagonal of R
                Q[:, k] = A[:, k] / norm  # Normalize the k-th column of A to get the k-th column of Q (unit vector)


        # When k > 1, A[:, k] has already been orthogonalized with respect to the first k - 1 columns of Q (i<k)
            for j in range(k+1, m):
                R[k, j] = np.dot(Q[:, k], A[:, j])
                A[:, j] = A[:, j] - R[k, j] * Q[:, k]

        return QRResult(Q, R)

    @staticmethod
    def _householder_reflection(A: NumericArray) -> QRResult:
        """
        Compute the QR decomposition of a matrix using Householder reflections.

        Parameters:
        A (np.ndarray): The input matrix.

        Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing the orthogonal matrix Q and the upper triangular matrix R.
        """
        n, m = A.shape
        Q = np.eye(n)
        R = A.copy()

        for i in range(min(m - 1, n)):  # range like this becuase we don't need to householder the last col
            x = R[i:, i]  # Reflecting vector i, the i-th column of R
            e = np.zeros_like(x)
            e[0] = np.copysign(np.linalg.norm(x), -x[0])  # Vector (-||x||, 0, ..., 0)
            u = x - e
            if np.linalg.norm(u) == 0:
                continue  # Avoid division by zero if u is zero
            v = u / np.linalg.norm(u)

            Q_i = np.eye(n)
            Q_i[i:, i:] -= 2 * np.outer(v, v) / np.dot(v, v)  # H=I-2vv^T/v^Tv

            R = Q_i @ R
            Q = Q @ Q_i.T

        return QRResult(Q, R)

    @staticmethod
    def _givens_rotation(A: NumericArray) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute the QR decomposition of a matrix using Givens rotations.

        Parameters:
        A (np.ndarray): The input matrix.

        Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing the orthogonal matrix Q and the upper triangular matrix R.
        """
        n, m = A.shape
        # Initialize Q as identity matrix of size n x n
        Q = np.eye(n)
        # Initialize R as a copy of A to avoid modifying the input
        R = A.copy()

        # Iterate over columns, left > right
        for j in range(n):
            for i in range(m - 1, j, -1):  # Interate over rows from the bottom (m-1), stopping at the row of the diagonal (j)
                if abs(R[i, j]) > 1e-12:
                    r = np.hypot(R[i-1, j], R[i, j])
                    c = R[i-1, j] / r
                    s = -R[i, j] / r

                    G = np.eye(m)
                    G[i-1:i+1, i-1:i+1] = np.array([[c, -s], [s, c]])

                    R = G @ R
                    Q = Q @ G.T

                ### ABOVE WE ARE DOING THIS ###
                    #     ... i ...
                    # ... [   |   ]
                    # j-1 [ a | * ]
                    # j   [ b | * ]
                    # ... [   |   ]

                    # Each Givens rotation zeroes out the lower element (b) by rotating it with the upper element (a)

                        # ... i ...
                    # ... [   |   ]
                    # j-1 [ a'| * ]
                    # j   [ 0 | * ]
                    # ... [   |   ]

                    # Where a' is the new value after rotation
                    # Gradually, we end up with an upper triangular R matrix

        return QRResult(Q, R)

    @staticmethod
    def reconstruct(Q: NumericArray, R: NumericArray) -> np.ndarray:
        """
        Reconstruct the original matrix from the QR decomposition.

        Parameters:
        Q (np.ndarray): The orthogonal matrix.
        R (np.ndarray): The upper triangular matrix.

        Returns:
        np.ndarray: The reconstructed matrix.
        """
        return Q @ R