import numpy as np

def generateIntegers(nIterations, nIntegers):
    out = np.random.randint(0, 10, (nIterations, nIntegers))
    for i in range(0, len(out)):
        for j in range(0, len(out[i,:])-1):
            if out[i,j] == out[i,j+1]:
                out[i,j+1] +=1
    return out