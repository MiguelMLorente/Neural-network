import numpy as np

def playOrderNumbersGame(numbers, network):
    networkOutput = network.forwardPass(numbers)
    scores = []

    for i in range(0, len(networkOutput)):
        idx = np.argsort(networkOutput[i,:])
        numbersOrderedByNetwork = numbers[i,idx]
        
        scores.append(getScore(numbers[i,:], numbersOrderedByNetwork))

    return np.mean(scores)

def getScore(originalNumbers, numbersOrderedByNetwork):
    orderedNumbers = np.sort(originalNumbers)
    
    leastSquares = 0
    realSquares = 0

    for i in range(0, len(numbersOrderedByNetwork)-1):
        leastSquares += (orderedNumbers[i] - orderedNumbers[i+1]) ** 2 
        realSquares += (numbersOrderedByNetwork[i] - numbersOrderedByNetwork[i+1]) ** 2
    return 100 * leastSquares / realSquares


def testPlayOrderNumbersGame(numbers, network):
    networkOutput = network.forwardPass(numbers)
    scores = []
    ordered = []

    for i in range(0, len(networkOutput)):
        idx = np.argsort(networkOutput[i,:])
        numbersOrderedByNetwork = numbers[i,idx]
        
        scores.append(getScore(numbers[i,:], numbersOrderedByNetwork))
        ordered.append(numbersOrderedByNetwork)

    return ordered, np.mean(scores)
