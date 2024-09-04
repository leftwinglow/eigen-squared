import numpy as np
from typing import NamedTuple

NumericArray = np.ndarray[tuple[int, ...], np.dtype[np.float64 | np.int_]]

class EigenpairsResult(NamedTuple):
    eigenvalues: NumericArray | float
    eigenvectors: NumericArray

class QRResult(NamedTuple):
    Q: np.ndarray
    R: np.ndarray

class CholeskyResult(NamedTuple):
    L: np.ndarray
    LT: np.ndarray

class LUResult(NamedTuple):
    L: np.ndarray
    U: np.ndarray

class GershgorinResult(NamedTuple):
    centers: np.ndarray
    radii: np.ndarray