"""
This is an implemention of CFR for Kuhn Poker. We see that the payoff is around -0.056, because
player 1 needs to act first. Player 2 is at an advantage because they act second.
"""
from audioop import avg
from distutils.log import info
from random import shuffle
import numpy as np

# Kuhn Poker definitions
PASS, BET, NUM_ACTIONS = 0,1,2
nodeMap = {}

class Node:
	def __init__(self) -> None:
		self.infoSet = ""
		self.regretSum = np.zeros(NUM_ACTIONS)
		self.strategy = np.zeros(NUM_ACTIONS)
		self.strategySum = np.zeros(NUM_ACTIONS)

	def getStrategy(self, realization_weight):
		for a in range(NUM_ACTIONS):
			self.strategy[a] = max(0, self.regretSum[a])
		
		normalizingSum = self.strategy.sum()
		for a in range(NUM_ACTIONS):
			if (normalizingSum > 0):
				self.strategy[a] /= normalizingSum
			else:
				self.strategy[a] = 1 / NUM_ACTIONS

			self.strategySum[a] += realization_weight * self.strategy[a]

		return self.strategy

	def getAverageStrategy(self):
		normalizingSum = self.strategySum.sum()
		avgStrategy = np.zeros(NUM_ACTIONS)
		for a in range(NUM_ACTIONS):
			if (normalizingSum > 0):
				avgStrategy[a] = self.strategySum[a] / normalizingSum
			else:
				avgStrategy[a] = 1 / NUM_ACTIONS

		return avgStrategy

def cfr(cards, history, p0, p1):
	# print(history)
	plays = len(history)
	player = plays % 2
	opponent = 1 - player
	
	# Return payoff for terminal states
	if (plays > 1):
		terminalPass = history[plays-1] == 'p'
		doubleBet = history[plays-2:plays] == "bb"
		isPlayerCardHigher = cards[player] > cards[opponent]
		
		if terminalPass:
			if history == "pp": 
				return 1 if isPlayerCardHigher else -1
			else: 
				return 1
	
		elif doubleBet:
			return 2 if isPlayerCardHigher else -2
			
	infoSet = str(cards[player]) + history
	# Get information set node or create it if nonexistant
	if infoSet not in nodeMap:
		node = Node()
		node.infoSet = infoSet
		nodeMap[infoSet] = node
	else:
		node = nodeMap[infoSet]

	# For each action, recursively call cfr with additional history and probability
	strategy = node.getStrategy(p0 if player == 0 else p1)
	util = np.zeros(NUM_ACTIONS)
	nodeUtil = 0
	for a in range(NUM_ACTIONS):
		nextHistory = history + ("p" if a == 0 else "b")
		util[a] = - cfr(cards, nextHistory, p0 * strategy[a], p1) if player == 0 else - cfr(cards, nextHistory, p0, p1 * strategy[a])
		nodeUtil += strategy[a] * util[a]
	
	# For each action, compute and accumulate counterfactual regret
	for a in range(NUM_ACTIONS):
		regret = util[a] - nodeUtil
		node.regretSum[a] += (p1 if player == 0 else p0) * regret
	return nodeUtil

def train(iterations):
	cards = [1,2,3]
	util = 0
	for i in range(iterations):
		shuffle(cards)
		util += cfr(cards, "", 1,1)
		if (i % 100000 == 0):
			print("Average game value: ", util/i)
	
	
if __name__ == "__main__":
	train(1000000)