import numpy as np
import math

from NeuralNetworks.NeuralNetwork import NeuralNetwork
from NSGAII.NeuralNetworkMutation import crossNeuralNetworks
from NSGAII.NeuralNetworkMutation import mutateNeuralNetwork


class NSGAII:

    def __init__(self, nObj, nVar, popSize, nGen, objectiveEvaluator, lBound = -50, uBound = 50):

        self.nObjectives = nObj
        self.nVariables = nVar
        self.populationSize = popSize
        self.nGenerations = nGen

        self.lowerBounds = np.ones(self.populationSize) * lBound
        self.upperBounds = np.ones(self.populationSize) * uBound

        self.generationsRecord = []
        self.objectivesRecord = []
        self.frontRecord = []

        self.evaluateObjectives = lambda population: objectiveEvaluator(population)

    def initializePopulation(self, topology, actFunction): 
        
        # self.currentGeneration is a list of NeuralNetwork, each representing a
        # different individual 

        self.networkTopology = topology
        self.activationFunction = actFunction
        # Create a population of neural networks randomly generated
        self.currentGeneration = []
        for i in range(self.populationSize):
            self.currentGeneration.append(NeuralNetwork(topology, actFunction))
            
        self.generationId = 1

        self.currentObjectives = self.evaluateObjectives(self.currentGeneration)
         
        self.generationsRecord.append(self.currentGeneration)
        self.objectivesRecord.append(self.currentObjectives)
        
    def nonDominationSort(self):
        
        nIndividualDominatingThisOne = np.zeros(self.populationSize)
        dominatedIndividualsList = []
    
        front = 0
        F = [[]]
        
        self.individualAtFront = np.zeros(self.populationSize)
        
        for i in range(0,self.populationSize):

            dominatedIndividualsList.append([])
            
            for j in range(0, self.populationSize):
                
                dominating = 0
                dominated = 0
                
                for k in range(0, self.nObjectives):
                    if (self.currentObjectives[i,k] < self.currentObjectives[j,k]):
                        dominating += 1
                    elif (self.currentObjectives[i,k] > self.currentObjectives[j,k]):
                        dominated += 1
                
                if (dominating == 0 and dominated > 0):
                    nIndividualDominatingThisOne[i] +=  1
                elif (dominating > 0 and dominated == 0):
                    dominatedIndividualsList[i].append(j)
                
            if (nIndividualDominatingThisOne[i] == 0):
                F[front].append(i)
                    
        
        # Basic explanation of the code below: segmentation of the individuals in several 
        # layers according to their dominance. The first layer F[0] contains the individuals
        # that are not dominated by any other, i.e. nIndividualDominatingThisOne[i] = 0.
        
        # Go along the individuals contained in F[0], and whoever they dominate, reduce their
        # counter nIndividualDominatingThisOne by 1 (for every individual in F[0] that dominates
        # him). Then, if their counter comes to zero, add them into the next layer F[1].
        
        # Iterate until there are no added individuals.
        
        while (len(F[front]) > 0):
            Q = []
            for i in range(0, len(F[front])):
                for j in range(0, len(dominatedIndividualsList[F[front][i]])):
                    nIndividualDominatingThisOne[dominatedIndividualsList[F[front][i]][j]] -= 1
                    if (nIndividualDominatingThisOne[dominatedIndividualsList[F[front][i]][j]] == 0):
                        self.individualAtFront[dominatedIndividualsList[F[front][i]][j]] = front + 1
                        Q.append(dominatedIndividualsList[F[front][i]][j])
            
            front += 1
            F.append(Q)
        F.pop()
        
        self.crowdingDistance = np.zeros(self.populationSize)
        
        
        
        for front in range (0,len(F)):  
            
            for i in range(0,self.nObjectives):
                
                # Sorting the individuals in this front by their i-th objective
                idx = np.argsort(self.currentObjectives[F[front] , i])
                
                # Assign the limit cases an infinite value (maximum preference)
                self.crowdingDistance[F[front][idx[0]]] = np.inf
                self.crowdingDistance[F[front][idx[-1]]] = np.inf
                
                span = max(1e-6,self.currentObjectives[F[front][idx[-1]] , i] - self.currentObjectives[F[front][idx[0]] , i])
                                                                                             
                for k in range(1,len(idx)-1):
                    self.crowdingDistance[F[front][idx[k]]] += (self.currentObjectives[F[front][idx[k+1]] , i] - self.currentObjectives[F[front][idx[k-1]] , i])/span

    def individualSelection(self, proportion = 0.5):
        
        pool = np.round(proportion * self.populationSize)
        poolCandidates = list(range(0, self.populationSize))
        remainingCandidates = len(poolCandidates)
        
        selectedCandidates = []
        
        # Until the pool of selected candidates is filled, one pair will be extracted,
        # from which the best will be selected. The other candidate will return and 
        # can be selected again for a new comparison.
        
        while (pool > 0):
            
            candidate1 = np.random.randint(0,remainingCandidates)
            candidate2 = np.random.randint(0,remainingCandidates-1)
            if (candidate2>=candidate1):
                candidate2 += 1
                
            if (self.individualAtFront[poolCandidates[candidate1]] < self.individualAtFront[poolCandidates[candidate2]]):
                selectedCandidates.append(poolCandidates[candidate1])
                poolCandidates.pop(candidate1)
            elif (self.individualAtFront[poolCandidates[candidate1]] > self.individualAtFront[poolCandidates[candidate2]]):
                selectedCandidates.append(poolCandidates[candidate2])
                poolCandidates.pop(candidate2)
            else:
                if (self.crowdingDistance[poolCandidates[candidate1]] > self.crowdingDistance[poolCandidates[candidate2]]):
                    selectedCandidates.append(poolCandidates[candidate1])
                    poolCandidates.pop(candidate1)
                else:
                    selectedCandidates.append(poolCandidates[candidate2])
                    poolCandidates.pop(candidate2)
            pool -= 1
            remainingCandidates -= 1
            
        return selectedCandidates, len(selectedCandidates)
               
    def newGeneration(self, mu_crossover = 20, mu_mutation = 20, crossover_probability = 0.9, isSingleObjective = False):
        if (not isSingleObjective):
            parents, nParents = self.individualSelection()
        else:
            parents, nParents = self.individualSelectionSingleObjective()
        
        children = []
        
        while(len(children) < self.populationSize):
            # Crossover between two random parents
            if (np.random.rand() < crossover_probability):
                # Random parent selection
                parent1 = np.random.randint(0,nParents)
                parent2 = np.random.randint(0,nParents-1)
                if (parent2>=parent1):
                    parent2 += 1

                child1, child2 = crossNeuralNetworks(mu_crossover, 
                                                     self.networkTopology, 
                                                     self.currentGeneration[parent1], 
                                                     self.currentGeneration[parent2])
                # Two slots available
                if (self.populationSize - len(children) >= 2):
                    children.append(child1)
                    children.append(child2)
                # One slot available: select random child
                elif (np.random.rand() > 0.5):
                    children.append(child1)
                else:
                    children.append(child2)
            
            # Mutation of a single parent
            else:
                parent = self.currentGeneration[parents[np.random.randint(0,nParents)]]
                child = mutateNeuralNetwork(mu_mutation, self.networkTopology, parent)
                # There is always at least one slot availble, so no need for checks
                children.append(child)
                
        self.currentGeneration = children    
                
    def runMultiObjectiveOptimizer(self):
  
        for i in range(0, self.nGenerations):

            self.nonDominationSort()
            self.frontRecord.append(self.individualAtFront)
            self.newGeneration(20*math.log(i+2), 20*math.log(i+2))
            
            self.currentObjectives = self.evaluateObjectives(self.currentGeneration)
            
            self.generationsRecord.append(self.currentGeneration)
            self.objectivesRecord.append(self.currentObjectives)
            print([str(i+1) + " of " + str(self.nGenerations)])
            
            self.generationId += 1

    def individualSelectionSingleObjective(self, proportion = 0.5):
        pool = np.round(proportion * self.populationSize)
        poolCandidates = list(range(0, self.populationSize))
        remainingCandidates = len(poolCandidates)
        
        selectedCandidates = []
        
        # Until the pool of selected candidates is filled, one pair will be extracted,
        # from which the best will be selected. The other candidate will return and 
        # can be selected again for a new comparison.
        
        while (pool > 0):
            
            candidate1 = np.random.randint(0,remainingCandidates)
            candidate2 = np.random.randint(0,remainingCandidates-1)
            if (candidate2>=candidate1):
                candidate2 += 1
                
            if (self.currentObjectives[candidate1] >= self.currentObjectives[candidate2]):
                selectedCandidates.append(poolCandidates[candidate1])
                poolCandidates.pop(candidate1)
            else:
                selectedCandidates.append(poolCandidates[candidate2])
                poolCandidates.pop(candidate2)
                
            pool -= 1
            remainingCandidates -= 1
            
        return selectedCandidates, len(selectedCandidates)

    def runSingleObjectiveOptimizer(self):

        # There is no need to make non-domination sort if there is only one objective
        for i in range(0, self.nGenerations):
            self.newGeneration(20*math.log(i+2), 20*math.log(i+2), isSingleObjective = True)
            
            self.currentObjectives = self.evaluateObjectives(self.currentGeneration)
            
            self.generationsRecord.append(self.currentGeneration)
            self.objectivesRecord.append(self.currentObjectives)
            print([str(i+1) + " of " + str(self.nGenerations)])
            
            self.generationId += 1
        
        best = 0
        idBest = (0, 0)
        for i in range(0, len(self.objectivesRecord)):
            maximumId = np.argmax(self.objectivesRecord[i])
            maximum = self.objectivesRecord[i][maximumId]
            print(maximum)
            if (maximum > best):
                best = maximum
                idBest = (i, maximumId)
        return self.generationsRecord[idBest[0]][idBest[1]], best, idBest
    