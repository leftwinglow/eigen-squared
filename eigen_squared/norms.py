import numpy as np
from enum import Enum
from eigen_squared.eigen_types import NumericArray
from typing import Any

class FrobeniusTypes(str, Enum):
    Standard = 'std'
    NoDiagonal = 'non_diag'
    Diagonal = 'diag'

class FrobeniusNorms:
    @staticmethod
    def frobenius_norm(A: NumericArray, type: FrobeniusTypes = FrobeniusTypes.Standard) -> np.floating[Any]:
        match type:
            case FrobeniusTypes.Standard:
                return FrobeniusNorms._standard_norm(A)
            case FrobeniusTypes.NoDiagonal:
                return FrobeniusNorms._non_diag_norm(A)
            case FrobeniusTypes.Diagonal:
                return FrobeniusNorms._diag_norm(A)
            case _:
                raise ValueError(f"Invalid Frobenius norm type. Choose from: {', '.join([t.value for t in FrobeniusTypes])}")

    @staticmethod
    def _standard_norm(A: NumericArray) -> np.floating[Any]:
        return np.linalg.norm(A, ord='fro')

    @staticmethod
    def _non_diag_norm(A: NumericArray) -> np.floating[Any]:
        return np.linalg.norm(A - np.diag(np.diag(A)), ord='fro')

    @staticmethod
    def _diag_norm(A: NumericArray) -> np.floating[Any]:
        return np.linalg.norm(np.diag(np.diag(A)), ord='fro')
