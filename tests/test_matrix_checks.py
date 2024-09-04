import pytest
import numpy as np
from eigen_squared.matrix_checks import (
    is_vector, is_square, is_invertible, is_symmetric, is_hermitian,
    is_positive_definite, is_diagonally_dominant, is_diagonal,
    InvertibleMethods, PosDefMethods
)

# Test data
vector = np.array([1, 2, 3])
square_matrix = np.array([[1, 2], [3, 4]])
non_square_matrix = np.array([[1, 2, 3], [4, 5, 6]])
invertible_matrix = np.array([[1, 2], [3, 4]])
non_invertible_matrix = np.array([[1, 2], [2, 4]])
symmetric_matrix = np.array([[1, 2], [2, 1]])
non_symmetric_matrix = np.array([[1, 2], [3, 4]])
hermitian_matrix = np.array([[1, 2-1j], [2+1j, 3]])
non_hermitian_matrix = np.array([[1, 2+1j], [2+1j, 3]])
positive_definite_matrix = np.array([[2, -1], [-1, 2]])
non_positive_definite_matrix = np.array([[1, 2], [2, 1]])
diagonally_dominant_matrix = np.array([[3, -1, 1], [1, 3, -1], [-1, 1, 3]])
non_diagonally_dominant_matrix = np.array([[1, 2], [2, 1]])
diagonal_matrix = np.array([[1, 0], [0, 2]])
non_diagonal_matrix = np.array([[1, 2], [0, 2]])

def test_is_vector():
    assert is_vector(vector)
    assert not is_vector(square_matrix)

def test_is_square():
    assert is_square(square_matrix)
    assert not is_square(non_square_matrix)

@pytest.mark.parametrize("method", [InvertibleMethods.determinant, InvertibleMethods.rank])
def test_is_invertible(method):
    assert is_invertible(invertible_matrix, method)
    assert not is_invertible(non_invertible_matrix, method)

def test_is_symmetric():
    assert is_symmetric(symmetric_matrix)
    assert not is_symmetric(non_symmetric_matrix)
    with pytest.raises(ValueError):
        is_symmetric(non_square_matrix)

def test_is_hermitian():
    assert is_hermitian(hermitian_matrix)
    assert not is_hermitian(non_hermitian_matrix)
    with pytest.raises(ValueError):
        is_hermitian(non_square_matrix)

def test_is_positive_definite():
    assert is_positive_definite(positive_definite_matrix, PosDefMethods.strict_diagonally_dominant)
    assert not is_positive_definite(non_positive_definite_matrix, PosDefMethods.strict_diagonally_dominant)
    with pytest.raises(ValueError):
        is_positive_definite(non_square_matrix)
    with pytest.raises(ValueError):
        is_positive_definite(positive_definite_matrix, "invalid_method")  # type: ignore

@pytest.mark.parametrize("direction", ["row", "column"])
@pytest.mark.parametrize("strictness", ["strict", "weak"])
def test_is_diagonally_dominant(direction, strictness):
    assert is_diagonally_dominant(diagonally_dominant_matrix, direction, strictness)
    assert not is_diagonally_dominant(non_diagonally_dominant_matrix, direction, strictness)

def test_is_diagonal():
    assert is_diagonal(diagonal_matrix)
    assert not is_diagonal(non_diagonal_matrix)
