import numpy as np
from NeuralNetworks.Utils.NeuralLayer import neuralLayer
from IPython.display import clear_output
import matplotlib.pyplot as plt
import time

# NEURAL NETWORK CREATION

class NeuralNetwork:
    def __init__(self, topology, actFunction):
        self.network = []
        for layer in range(0, len(topology) - 1):
            self.network.append(neuralLayer(topology[layer], topology[layer + 1], actFunction))

    def forwardPass(self, inputData, train = False):
        out = [(None, inputData)]
        for layer in range(0, len(self.network)):
            
            z = out[-1][1] @ self.network[layer].w + self.network[layer].b
            a = self.network[layer].actFunction[0](z)
            
            out.append((z, a))
        
        if train:
            return out
        else:
            return out[-1][1]


    def _trainNeuralNetwork(self, inputData, expectedOutput, costFunction, learnRate):
        out = self.forwardPass(inputData, True)
        
        # Backwards pass
        deltas = []
        
        for l in reversed(range(0, len(self.network))):
            z = out[l+1][0]
            a = out[l+1][1]

            if (l == len(self.network) - 1):
                deltas.insert(0, costFunction[1](a, expectedOutput) * self.network[l].actFunction[1](a))
            else:
                deltas.insert(0, deltas[0] @ _w.T * self.network[l].actFunction[1](a))
            _w = self.network[l].w

            # Gradient descent
            self.network[l].b -= np.mean(deltas[0], axis = 0, keepdims = True) * learnRate
            self.network[l].w -= out[l][1].T @ deltas[0] * learnRate
                
        return out[-1][1]

    def train(self, nIterations, inputData, expectedOutput, costFunction, learningRate= 0.025, breakTolerance = 1e-4, plotCostFunction = False, plotLoss = False, plotMesh = False, plotStep = 25):
        loss = []
        for i in range (nIterations):
            pY = self._trainNeuralNetwork(inputData, expectedOutput, costFunction, learningRate)
            if (plotLoss and i % plotStep == 0):
                loss.append(costFunction[0](pY, expectedOutput))
                plt.semilogy(range(len(loss)), loss)
                plt.show()
                time.sleep(0.5)
                
            if (plotMesh and i % plotStep == 0):
                self.show(inputData, expectedOutput)
                
            if (loss[-1] <= breakTolerance and i % plotStep == 0):
                break
                
    def show(self, inputData, expectedOutput):
        resolution = 50
                
        _x0 = np.linspace(-1.5, 1.5, resolution)
        _x1 = np.linspace(-1.5, 1.5, resolution)
        _Y = np.zeros((resolution, resolution))

        for i0, x0 in enumerate(_x0):
            for i1, x1 in enumerate(_x1):
                _Y[i0,i1] = self.forwardPass(np.array([[x0, x1]]))
                
        plt.pcolormesh(_x0, _x1, _Y, cmap="coolwarm")
        plt.axis("equal")
        plt.scatter(inputData[expectedOutput[:,0] == 0, 0], inputData[expectedOutput[:,0] == 0, 1], c = "blue")
        plt.scatter(inputData[expectedOutput[:,0] == 1, 0], inputData[expectedOutput[:,0] == 1, 1], c = "red")
        
        clear_output(wait = True)
        plt.show()
        
        