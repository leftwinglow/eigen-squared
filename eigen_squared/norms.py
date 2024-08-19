import numpy as np
from enum import Enum
from eigen_squared.eigen_types import NumericArray

class FrobeniusTypes(str, Enum):
    Standard = 'std'
    NoDiagonal = 'non_diag'
    Diagonal = 'diag'

class FrobeniusNorms:
    @staticmethod
    def frobenius_norm(A: NumericArray, type: FrobeniusTypes = "Standard") -> float:
        match type:
            case FrobeniusTypes.Standard:
                return FrobeniusNorms._standard_norm(A)
            case FrobeniusTypes.NoDiagonal:
                return FrobeniusNorms._non_diag_norm(A)
            case FrobeniusTypes.Diagonal:
                return FrobeniusNorms._diag_norm(A)
            case _:
                raise ValueError(f"Invalid Frobenius norm type. Choose from: {', '.join([t.value for t in FrobeniusTypes])}")

    def _standard_norm(A: NumericArray) -> float:
        return np.linalg.norm(A, ord='fro')

    def _non_diag_norm(A: NumericArray) -> float:
        return np.linalg.norm(A - np.diag(np.diag(A)), ord='fro')

    def _diag_norm(A: NumericArray) -> float:
        return np.linalg.norm(np.diag(np.diag(A)), ord='fro')
