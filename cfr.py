import numpy as np


ROCK, PAPER, SCISSORS, NUM_ACTIONS = 0,1,2,3
regretSum = [0,0,0]
strategy = [0,0,0]
strategySum = [0,0,0]
oppStrategy = [0.4, 0.3, 0.3]

def getStrategy():
	normalizingSum = 0
	for i in range(NUM_ACTIONS):
		strategy[i] = max(regretSum[i], 0)
		normalizingSum += strategy[i]
		
	for i in range(NUM_ACTIONS):
		if normalizingSum > 0:
			strategy[i] /= normalizingSum
		else:
			strategy[i] = 1/NUM_ACTIONS
		
		strategySum[i] += strategy[i]
	
	return strategy

def getAction():
	r = np.random.random()
	cumulativeProbability = 0
	for i in range(NUM_ACTIONS - 1):
		cumulativeProbability += strategy[i]
		if (r < cumulativeProbability): break
	
	return i
		
		
def train(iterations: int):
	actionUtility = np.array()
		
		
