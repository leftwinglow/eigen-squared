import numpy as np
from eigen_squared import matrix_checks

N = 5
b = np.random.randint(-2000, 2000, size=(N, N))
b_symm = (b + b.T) / 2
c = b_symm.T @ b_symm

def test_is_square():
    assert matrix_checks.is_square(b), "Matrix is not square."

def test_is_symmetric():
    assert matrix_checks.is_symmetric(b_symm), "Matrix is not symmetric."

def test_strict_row_dominance():
    A = np.array([
        [5, 1, 1],
        [1, 6, 1],
        [1, 1, 7]
    ])
    assert matrix_checks.is_diagonally_dominant(A, 'row', 'weak'), "Matrix is not weakly column dominant."
    assert matrix_checks.is_diagonally_dominant(A, 'row', 'strict'), "Matrix is not strictly row dominant."

def test_strict_col_dominance():
    A = np.array([
        [5, 1, 1],
        [1, 6, 1],
        [1, 1, 7]
    ])
    assert matrix_checks.is_diagonally_dominant(A, 'column', 'weak'), "Matrix is not weakly column dominant."
    assert matrix_checks.is_diagonally_dominant(A, 'column', 'strict'), "Matrix is not strictly column dominant."

def test_weak_row_dominance():
    A = np.array([
        [5, 2, 3],
        [2, 5, 3],
        [2, 3, 5]
    ])
    assert matrix_checks.is_diagonally_dominant(A, 'row', 'strict') is False
    assert matrix_checks.is_diagonally_dominant(A, 'row', 'weak'), "Matrix is not weakly row dominant."


def test_weak_col_dominance():
    A = np.array([
        [5, 3, 4],
        [2, 6, 3],
        [3, 3, 7]
    ])
    assert matrix_checks.is_diagonally_dominant(A, 'column', 'strict') is False
    assert matrix_checks.is_diagonally_dominant(A, 'column', 'weak'), "Matrix is not weakly column dominant."