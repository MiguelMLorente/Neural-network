from NeuralNetworks.NeuralNetwork import NeuralNetwork
from NeuralNetworks.Utils.CostFunction import CostFunction
from NeuralNetworks.Utils.ActivationFunction import ActivationFunction
from DataSets.GenerateCircles import makeSingleCircleDataSet
from NSGAII.NSGAII import NSGAII
import numpy as np

# Create a single set of points within two circles with some random noise.
# Points within the outer circle will be marked with a '0' in the variable
# circle, while those in the inner circle will be marked with a '1'. The 
# variable points will contain their 'x' and 'y' coordinate.


# Define the game that the AI will be learning to play. Such game will define
# an evaluation function, as well as the ammount of commands that the AI can 
# execute



# Number of neurons at the first layer equals the dimension of the inputData,
# while the number of neurons at the last layer must match the dimension of
# the output data

nInputVariables = 2
nOutputVariables = 1

# The topology represents the number of neurons at each layer
topology = [nInputVariables, 3, nOutputVariables]

# Create the network
nIndividuals = 3
nGenerations = 4

optimizer = NSGAII(nObj = nOutputVariables,
                   nVar = nInputVariables,
                   popSize = nIndividuals,
                   nGen = nGenerations,
                   objectiveEvaluator = lambda x: np.random.rand(nIndividuals, nOutputVariables),
                   lBound = -50,
                   uBound = 50)
optimizer.initializePopulation(topology, ActivationFunction.sigmoid.value)
optimizer.runOptimizer()

# Train the network with the same data for a certain maximum number of
# iterations and a given learning rate

