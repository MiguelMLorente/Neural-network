import numpy as np

# CLASE DE LA CAPA DE LA RED

class neuralLayer():
    def __init__(self, connections, neurons, actFunction):
        self.actFunction = actFunction
        self.b = np.random.rand(1, neurons) * 2 - 1
        self.w = np.random.rand(connections, neurons) * 2 - 1