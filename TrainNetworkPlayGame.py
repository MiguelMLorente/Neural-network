from NeuralNetworks.NeuralNetwork import NeuralNetwork
from NeuralNetworks.Utils.ActivationFunction import ActivationFunction
from DataSets.GenerateIntegers import generateIntegers
from NSGAII.NSGAII import NSGAII
from GamePlay.OrderNumbers import playOrderNumbersGame
from GamePlay.OrderNumbers import testPlayOrderNumbersGame
import numpy as np


np.seterr(all='raise')

# Define the game that the AI will be learning to play. Such game will define
# an evaluation function, as well as the ammount of commands that the AI can 
# execute

nTrainingCases = 10000
nTestCases= 10
nIntegers = 3

def playGame(population):
    dataSet = generateIntegers(nTrainingCases, nIntegers)
    scores = []
    for individual in population:
        scores.append(playOrderNumbersGame(dataSet, individual))
    return np.array(scores)

def testGame(network, nTestCases = 10):
    dataSet = generateIntegers(nTestCases, nIntegers)

    orderedNum, score = testPlayOrderNumbersGame(dataSet, network)
    return score, dataSet, orderedNum




# Number of neurons at the first layer equals the dimension of the inputData,
# while the number of neurons at the last layer must match the dimension of
# the output data. Such numbers must be equal to the ammount of integers that
# we are aiming to order

nInputVariables = nIntegers
nOutputVariables = nIntegers

# The topology represents the number of neurons at each layer
topology = [nInputVariables, 8, 12, 8, nOutputVariables]

# Create the network
nIndividuals = 50
nGenerations = 200

optimizer = NSGAII(nObj = nOutputVariables,
                   nVar = nInputVariables,
                   popSize = nIndividuals,
                   nGen = nGenerations,
                   objectiveEvaluator = lambda population: playGame(population),
                   lBound = -50,
                   uBound = 50)
optimizer.initializePopulation(topology, ActivationFunction.sigmoid.value)
optimizedNetwork, fitness, idx = optimizer.runSingleObjectiveOptimizer()

testFitness, testDataSet, testResults = testGame(optimizedNetwork, nTestCases)

# Train the network with the same data for a certain maximum number of
# iterations and a given learning rate

