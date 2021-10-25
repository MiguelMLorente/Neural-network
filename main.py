import numpy as np
import matplotlib.pyplot as plt
import time
from IPython.display import clear_output

from NeuralNetworks.NeuralNetwork import *
from NeuralNetworks.Utils.CostFunction import CostFunction
from NeuralNetworks.Utils.ActivationFunction import ActivationFunction
from DataSets.GenerateCircles import makeSingleCircleDataSet

#CREAR EL DATASET

n = 500
p = 2

points, circle = makeSingleCircleDataSet(n)

# FUNCIONES DE ACTIVACION
        
sigm = ActivationFunction.sigmoid.value

# COST FUNCTION

l2_cost = CostFunction.leastSquares.value

# LAUNCH TRAINING ALGORITHM

topology = [p, 4, 8, 4, 1]
neuralNetwork = createNeuralNetwork(topology, sigm)

loss = []
step = 25

for i in range (5000):
    pY = trainNeuralNetwork(neuralNetwork, points, circle, l2_cost, 0.025)
    if (i % step == 0) :
        loss.append(l2_cost[0](pY, circle))
        
        res = 50
        
        _x0 = np.linspace(-1.5, 1.5, res)
        _x1 = np.linspace(-1.5, 1.5, res)
        _Y = np.zeros((res,res))

        for i0, x0 in enumerate(_x0):
            for i1, x1 in enumerate(_x1):
                _Y[i0,i1] = trainNeuralNetwork(neuralNetwork, np.array([[x0, x1]]), circle, l2_cost, 0.5, False)
                
        plt.pcolormesh(_x0, _x1, _Y, cmap="coolwarm")
        plt.axis("equal")
        plt.scatter(points[circle[:,0] == 0, 0], points[circle[:,0] == 0, 1], c = "blue")
        plt.scatter(points[circle[:,0] == 1, 0], points[circle[:,0] == 1, 1], c = "red")
        
        clear_output(wait = True)
        plt.show()
        
        plt.plot(range(len(loss)), loss)
        plt.show()
        
        time.sleep(0.5)
