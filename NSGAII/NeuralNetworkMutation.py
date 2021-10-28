import numpy as np
import math

def getNewBeta(mu_crossover):
    u = np.random.rand()
    if (u <= 0.5):
        beta = math.pow(2*u, 1/(mu_crossover+1))
    else:
        beta = math.pow(2*(1-u), -1/(mu_crossover+1))
    return beta

def getBetaList(mu_c, length):
    out = []
    for i in range(0, length):
        out.append(getNewBeta(mu_c))
    return out
        
def getBetaMatrix(mu_c, rows, cols):
    out = []
    for i in range(0, rows):
        out.append(getBetaList(mu_c, cols))
    return out


def crossNeuralNetworks(mu_c, topology, parent1, parent2):
    halfBetaPlus1 = ([], [])
    halfBetaMinus1 = ([], [])

    for layer in range(0, len(topology) - 1):
        beta_b = np.array(getBetaList(mu_c, topology[layer + 1]))
        beta_w = np.array(getBetaMatrix(mu_c, topology[layer], topology[layer + 1]))
        
        halfBetaPlus1[0].append(0.5 * (beta_b + 1))
        halfBetaPlus1[1].append(0.5 * (beta_w + 1))
        halfBetaMinus1[0].append(0.5 * (beta_b - 1))
        halfBetaMinus1[1].append(0.5 * (beta_w - 1))
    
    child1 = parent1 * halfBetaPlus1 + parent2 * halfBetaMinus1
    child2 = parent1 * halfBetaMinus1 + parent2 * halfBetaPlus1
    
    return child1, child2




def getNewDelta(mu_mutation):
    r = np.random.rand()
    if (r < 0.5):
        delta = math.pow(2*r, 1/(mu_mutation+1)) - 1
    else:
        delta = 1 - math.pow(2*(1-r), 1/(mu_mutation+1))
    return delta

def getDeltaList(mu_m, length):
    out = []
    for i in range(0, length):
        out.append(getNewDelta(mu_m))
    return out
        
def getDeltaMatrix(mu_m, rows, cols):
    out = []
    for i in range(0, rows):
        out.append(getDeltaList(mu_m, cols))
    return out

def mutateNeuralNetwork(mu_m, topology, parent):
    deltaPlus1 = ([], [])

    for layer in range(0, len(topology) - 1):
        deltaPlus1[0].append(np.array(getDeltaList(mu_m, topology[layer + 1])) + 1)
        deltaPlus1[0].append(np.array(getDeltaMatrix(mu_m, topology[layer], topology[layer + 1])) + 1)

    child = parent * deltaPlus1
    
    return child