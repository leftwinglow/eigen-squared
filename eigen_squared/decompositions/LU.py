import numpy as np
from enum import Enum
from eigen_squared.eigen_types import NumericArray, LUResult

class LUMethods(str, Enum):
    doolittle = "Doolittle"

class LUDecomposition:
    @staticmethod
    def decompose(A: NumericArray, method: LUMethods = LUMethods.doolittle) -> LUResult:
        match method:
            case LUMethods.doolittle:
                L, U = LUDecomposition._LU_Doolittle(A)

        return LUResult(L, U)

    @staticmethod
    def _LU_Doolittle(A: NumericArray) -> tuple[np.ndarray, np.ndarray]:
        n = A.shape[0]
