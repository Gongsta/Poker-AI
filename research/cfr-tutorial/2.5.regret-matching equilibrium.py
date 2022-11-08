# Regret Matching Algorithm Implemented from this paper: http://modelai.gettysburg.edu/2013/cfr/cfr.pdf
from encodings import normalize_encoding
import numpy as np

ROCK, PAPER, SCISSORS, NUM_ACTIONS = 0,1,2,3
regretSum = np.zeros(NUM_ACTIONS)
oppRegretSum= np.zeros(NUM_ACTIONS)
strategy = np.zeros(NUM_ACTIONS)
strategySum = np.zeros(NUM_ACTIONS)
oppStrategySum = np.zeros(NUM_ACTIONS)
oppStrategy = np.array([1, 0, 0])
strategy_hist = []
oppStrategy_hist = []

def getStrategy():
	normalizingSum = 0
	for a in range(NUM_ACTIONS):
		strategy[a] = max(regretSum[a], 0)
		normalizingSum += strategy[a]
		
	for a in range(NUM_ACTIONS):
		if normalizingSum > 0:
			strategy[a] /= normalizingSum
		else:
			strategy[a] = 1/NUM_ACTIONS
		
		strategySum[a] += strategy[a]
	
	return strategy

def getOppStrategy():
	normalizingSum = 0
	for a in range(NUM_ACTIONS):
		oppStrategy[a] = max(oppRegretSum[a], 0)
		normalizingSum += oppStrategy[a]
		
	for a in range(NUM_ACTIONS):
		if normalizingSum > 0:
			oppStrategy[a] /= normalizingSum
		else:
			oppStrategy[a] = 1/NUM_ACTIONS
		
		oppStrategySum[a] += oppStrategy[a]
	
	return strategy

def getAction(strategy):
	r = np.random.random()
	cumulativeProbability = 0
	action = 0
	for a in range(NUM_ACTIONS):
		action = a
		cumulativeProbability += strategy[a]
		if (r < cumulativeProbability): break
	
	return action
		
		
def train(iterations: int): 
	actionUtility = np.zeros(NUM_ACTIONS)
	 
	for i in range(iterations):
		# Get regret-matched mixed-strategy actions
		strategy = getStrategy()
		myAction = getAction(strategy)
		oppAction = getAction(oppStrategy)
		
		# Compute action utilities
		actionUtility[oppAction] = 0
		actionUtility[(oppAction + 1) % NUM_ACTIONS] = 1
		actionUtility[(oppAction - 1) % NUM_ACTIONS] = -1
		
		# Accumulate action regrets
		for a in range(NUM_ACTIONS):
			regretSum[a] += actionUtility[a] - actionUtility[myAction]
	
		
def oppTrain(iterations: int):
	oppActionUtility = np.zeros(NUM_ACTIONS)
	 
	for i in range(iterations):
		# Get regret-matched mixed-strategy actions
		strategy = getStrategy()
		oppStrategy = getOppStrategy()
		myAction = getAction(strategy)
		oppAction = getAction(oppStrategy)
		
		oppActionUtility[myAction] = 0
		oppActionUtility[(myAction + 1) % NUM_ACTIONS] = 1
		oppActionUtility[(myAction - 1) % NUM_ACTIONS] = -1
		# Accumulate action regrets
		for a in range(NUM_ACTIONS):
			oppRegretSum[a] += oppActionUtility[a] - oppActionUtility[oppAction]
	
def trainOverall(iterations: int):
	for _ in range(iterations):
		global regretSum, oppRegretSum
		# Reset the regrets
		regretSum = np.zeros(NUM_ACTIONS)
		oppRegretSum= np.zeros(NUM_ACTIONS)
		train(1000) # Train your regret algorithm
		oppTrain(1000) # Train your opponent regret algorithm
	
def getAverageStrategy(strategySum):
	avgStrategy = np.zeros(NUM_ACTIONS)
	normalizingSum = 0
	for a in range(NUM_ACTIONS):
		normalizingSum += strategySum[a]

	for a in range(NUM_ACTIONS):
		if (normalizingSum > 0):
			avgStrategy[a] = strategySum[a] / normalizingSum
		else:
			avgStrategy[a] = 1 / NUM_ACTIONS

	return avgStrategy

def main():
	trainOverall(100)
	avgStrategy = getAverageStrategy(strategySum)
	print("Strategy computed: ", avgStrategy)
	print("Sum of the probabilities should be 1: ", avgStrategy.sum())
	oppAvgStrategy = getAverageStrategy(oppStrategySum)
	print("Opponent Strategy computed: ", oppAvgStrategy)
	print("Sum of the probabilities should be 1: ", oppAvgStrategy.sum())
	
if __name__ == "__main__":
	main()