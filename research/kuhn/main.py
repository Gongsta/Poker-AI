# Use this script to play against the trained AI that can play Kuhn Poker Near Nash Equilibrium
from random import shuffle
import joblib
import numpy as np

from research.kuhn.cfr import Node

PLAYER = 0
AI = 1

def getAction(strategy):
	r = np.random.random()
	cumulativeProbability = 0
	action = 0
	for a in range(len(strategy)):
		action = a
		cumulativeProbability += strategy[a]
		if (r < cumulativeProbability): break
	
	if action == 0: 
		return 'p'
	else:
		return 'b'
		
def terminal(history):
	if (len(history) > 1) and (history[-1] == 'p' or history[-2:] == "bb"): 
		return True
	else: 
		return False

if __name__ == "__main__":
	score = [0, 0] # [PLAYER_SCORE, AI_SCORE]
	# Load the nodeMap
	nodeMap: Node = joblib.load("KuhnNodeMap.joblib")
	
	first_player_to_move = 0

	cards = [1,2,3] # index 0 is for PLAYER, index 1 is for AI
	while True:
		# Setup a new round
		history = ""
		first_player_to_move += 1 # Alternate players to play each round
		first_player_to_move %= 2
		player_to_move = first_player_to_move
		shuffle(cards)

		print("You have been dealt a:", cards[0])
		if player_to_move == PLAYER:
			action = input('Please decide whether to pass or bet ("p" or "b")')
		else:
			action = getAction(nodeMap[str(cards[1])].getAverageStrategy())
			print("Your opponent has decided to play:", action)

		history += action

		while not terminal(history):
			plays = len(history)
			player = plays % 2
			
			if player == 0:
				action = input('Please decide whether to pass or bet ("p" or "b")')
			else:
				action = getAction(nodeMap[str(cards[1]) + history].getAverageStrategy())
			
			history += action
		# Return payoff for terminal states
		if (plays > 1):
			terminalPass = history[-1] == 'p'
			doubleBet = history[-2:] == "bb"
			isPlayerCardHigher = cards[0] > cards[1]
			
			if terminalPass:
				if history == "pp": 
					if isPlayerCardHigher:
						score[0] += 1
						score[1] -= 1
					
					else:
						score[0] += 1
						score[1] -= 1
				else: # Equivalent to folding
					score[len(history) % 2] += 1
					score[(first_player_to_move + 1)] -= 1
		
			elif doubleBet:
				if isPlayerCardHigher:
					score[0] += 2
					score[1] -= 2
				
				else: 
					score[0] -= 2
					score[1] += 2

		
		

	
	
