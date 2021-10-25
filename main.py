from NeuralNetworks.NeuralNetwork import NeuralNetwork
from NeuralNetworks.Utils.CostFunction import CostFunction
from NeuralNetworks.Utils.ActivationFunction import ActivationFunction
from DataSets.GenerateCircles import makeSingleCircleDataSet

# Create a single set of points within two circles with some random noise.
# Points within the outer circle will be marked with a '0' in the variable
# circle, while those in the inner circle will be marked with a '1'. The 
# variable points will contain their 'x' and 'y' coordinate.

n = 500
points, circle = makeSingleCircleDataSet(n)

# Number of neurons at the first layer equals the dimension of the inputData,
# while the number of neurons at the last layer must match the dimension of
# the output data

nInputVariables = len(points[0])
nOutputVariables = len(circle[0])

# The topology represents the number of neurons at each layer
topology = [nInputVariables, 4, 8, 4, nOutputVariables]

# Create the network
network = NeuralNetwork(topology, ActivationFunction.sigmoid.value)

# Train the network with the same data for a certain number of iterations and LR
nIterations = 100000
learningRate = 0.005

# Optional parameters for the training process visualization
plotLoss = True
plotMesh = True

network.train(nIterations, points, circle, CostFunction.leastSquares.value, plotLoss, plotMesh, learningRate, 2000)

network.show(points, circle)