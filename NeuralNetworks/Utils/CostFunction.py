from enum import Enum
import numpy as np

class CostFunction(Enum):
    leastSquares = (lambda Yp, Yr: np.mean((Yp - Yr) ** 2),
           lambda Yp, Yr: Yp - Yr)