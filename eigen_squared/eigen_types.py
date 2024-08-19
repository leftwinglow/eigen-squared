import numpy as np
from typing import NamedTuple

NumericArray = np.ndarray[np.float64 | np.int_]

class EigenpairsResult(NamedTuple):
    eigenvalues: NumericArray | float
    eigenvectors: NumericArray

# @dataclass
# class EigenpairsResult:
#     eigenvalues: Any
#     eigenvectors: Any

#     def __post_init__(self):
#         # Ensure numpy arrays for consistency
#         self.eigenvalues = np.array(self.eigenvalues)
#         self.eigenvectors = np.array(self.eigenvectors)

#         # Sort the eigenvalues and corresponding eigenvectors
#         indices = np.argsort(self.eigenvalues)
#         self.eigenvalues = self.eigenvalues[indices]
#         self.eigenvectors = self.eigenvectors[:, indices]

#     def __getitem__(self, index):
#         if index == 0:
#             return self.eigenvalues
#         elif index == 1:
#             return self.eigenvectors
#         else:
#             raise IndexError("Index out of range. Use 0 for eigenvalues and 1 for eigenvectors.")


class QRResult(NamedTuple):
    Q: np.ndarray
    R: np.ndarray

class CholeskyResult(NamedTuple):
    L: np.ndarray
    LT: np.ndarray