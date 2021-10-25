import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
import time
from IPython.display import clear_output
from sklearn.datasets import make_circles

#CREAR EL DATASET

n = 500
p = 2

points, circle = make_circles(n_samples = n, factor = 0.6, noise = 0.05)
#circle = circle.reshape(n,1)
plt.scatter(points[circle == 0, 0], points[circle == 0, 1], c = "blue")
plt.scatter(points[circle == 1, 0], points[circle == 1, 1], c = "red")
plt.axis("equal")
plt.show()
circle = circle.reshape(n,1)

# CLASE DE LA CAPA DE LA RED

class neuralLayer():
    def __init__(self, connections, neurons, actFunction):
        self.actFunction = actFunction
        self.b = np.random.rand(1, neurons) * 2 - 1
        self.w = np.random.rand(connections, neurons) * 2 - 1
        
        
# FUNCIONES DE ACTIVACION
        
sigm = (lambda x: 1 / (1 + np.e ** (-x)),
        lambda x: x* (1 - x))
relu = (lambda x: np.maximum(0, x),
        lambda x: 1 if (x > 0) else 0)

_x = np.linspace(-5, 5, 100)
plt.plot(_x, sigm[0](_x))
plt.show()
plt.plot(_x, relu[0](_x))
plt.show()


# NEURAL NETWORK CREATION

def createNeuralNetwork(topology, actFunction):
    nn = []
    for l, layer in enumerate(topology[:-1]):
        nn.append(neuralLayer(topology[l], topology[l+1], actFunction))
    return nn


# COST FUNCTION

l2_cost = (lambda Yp, Yr: np.mean((Yp - Yr) ** 2),
           lambda Yp, Yr: Yp - Yr)


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
        
        # plt.plot(range(len(loss)), loss)
        # plt.show()
        
        time.sleep(0.5)





