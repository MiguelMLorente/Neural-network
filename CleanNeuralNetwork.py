from Networks.NeuralNetwork import neuralNetwork
from Networks.ActivationFunction import activationFunction as actF
from Networks.CostFunction import costFunction as costF
from DataSets.DataProvider import DataProvider as provider
from DataSets.InvariantDataGenerator import InvariantDataGenerator as invariantGenerator
from DataSets.DataPool import DataPool as pool

nPoints = 500
p = 2

topology = [p, 2, 4, 8, 4, 1]
network = neuralNetwork(topology, actF.sigmoid.value)
dataProvider = provider(2000, invariantGenerator(pool.singleCircleData(nPoints)))
network.train2(0.01, True, True)

print(network.forwardPass([[0, 0], [10, 10]]))