import numpy as np

# CLASE DE LA CAPA DE LA RED

class neuralLayer():
    def __init__(self, connections = None, neurons = None, actFunction = None, isEmpty = False):
        if (not isEmpty):
            self.actFunction = actFunction
            self.b = np.random.rand(1, neurons) * 2 - 1
            self.w = np.random.rand(connections, neurons) * 2 - 1

        else:
            self.actFunction = actFunction
            self.b = []
            self.w = [[]]

    def __add__(self, other):
        output = neuralLayer(actFunction = self.actFunction, isEmpty = True)
        output.b = self.b + other.b
        output.w = self.w + other.w
        return output

    def __mul__(self, value):
        try:
            output = neuralLayer(actFunction = self.actFunction, isEmpty = True)
            if (type(value) is int):
                output.b = self.b * value
                output.w = self.w * value
            else:
                output.b = self.b * value[0]
                output.w = self.w * value[1]
            return output
        except:
            print(value)
