from enum import Enum
import numpy as np

class ActivationFunction(Enum):
    sigmoid = (lambda x: 1 / (1 + np.e ** (-x)),
        lambda x: x* (1 - x))
    relu = (lambda x: np.maximum(0, x),
        lambda x: 1 if (x > 0) else 0)