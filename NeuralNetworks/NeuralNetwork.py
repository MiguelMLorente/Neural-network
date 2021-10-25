import numpy as np



# CLASE DE LA CAPA DE LA RED

class neuralLayer():
    def __init__(self, connections, neurons, actFunction):
        self.actFunction = actFunction
        self.b = np.random.rand(1, neurons) * 2 - 1
        self.w = np.random.rand(connections, neurons) * 2 - 1


# NEURAL NETWORK CREATION

def createNeuralNetwork(topology, actFunction):
    nn = []
    for l, layer in enumerate(topology[:-1]):
        nn.append(neuralLayer(topology[l], topology[l+1], actFunction))
    return nn


# TRAINING NEURAL NETWORK FUNCTION

def trainNeuralNetwork(network, X, Y, l2_cost, lr = 0.5, train = True):
    out = [(None, X)]
    # Forward pass
    for l, layer in enumerate(network):
        
        z = out[-1][1] @ network[l].w + network[l].b
        a = network[l].actFunction[0](z)
        
        out.append((z, a))
    
    if train:
        # Backwards pass
        deltas = []
        
        for l in reversed(range(0, len(network))):
            z = out[l+1][0]
            a = out[l+1][1]

            if (l == len(network) - 1):
                deltas.insert(0, l2_cost[1](a, Y) * network[l].actFunction[1](a))
            else:
                deltas.insert(0, deltas[0] @ _w.T * network[l].actFunction[1](a))
            _w = network[l].w

            # Gradient descent
            print( np.mean(deltas[0], axis = 0, keepdims = True) * lr)
            network[l].b = network[l].b - np.mean(deltas[0], axis = 0, keepdims = True) * lr
            network[l].w = network[l].w - out[l][1].T @ deltas[0] * lr
            
    return out[-1][1]