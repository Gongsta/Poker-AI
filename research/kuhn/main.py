"""
Use this script to play against the trained Kuhn Poker AI either manually in real 
time from the terminal, or run simulations using a fixed strategy that you compute.


To run the simulation, simply run
---
python main.py
---

To play against the AI in the terminal, run:
---
python main.py --play
---
"""
from random import shuffle
import joblib
import numpy as np
import argparse
import matplotlib.pyplot as plt

from cfr import Node

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
		
def getStrategy(card, strategy=0):
	"""
	stragegy=0 -> Pass if you have 1, Bet if you have 3, and play 50% of your hands with 2
	stragegy=1 -> Always pass
	stragegy=2 -> Always bet
	strategy=3 -> CFR
	"""
	if strategy == 0:
		return getAction(nodeMap[str(card)].getAverageStrategy())
	elif strategy == 1:
		if card == 1:
			return 'p'
		elif card == 3:
			return 'b'
		else:
			r = np.random.random()
			if r <= 0.5:
				return 'p'
			else:
				return 'b'
	elif strategy == 2:
		return 'p'
	elif strategy == 3:
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

	parser = argparse.ArgumentParser(description='Play Kuhn Poker against the best AI possible.')
	parser.add_argument("-p", "--play",
                    action="store_true", dest="user_input", default=False,
                    help="Manually play against the AI through the terminal.")
	parser.add_argument("-v", "--verbose",
                    action="store_true", dest="verbose", default=False,
                    help="Manually play against the AI through the terminal.")

	args = parser.parse_args()
	user_input = args.user_input
	verbose = args.verbose # In case you want to see each game printed out in the terminal while running the simulation
	
	user_scores_over_time = []
	opponent_scores_over_time = []

	cards = [1,2,3] # index 0 is for PLAYER, index 1 is for AI
	for _ in range(1000000):
		# Setup a new round
		history = ""
		first_player_to_move += 1 # Alternate players to play each round
		first_player_to_move %= 2
		player_to_move = first_player_to_move
		shuffle(cards)

		if user_input or verbose:
			print("--------------------------")
			print("Current Scoreboard:")
			print("You: {}, Opponent: {}\n".format(score[0], score[1]))
			print("You have been dealt a:", cards[0])
		
		# Alternate every round between the players playing first
		if player_to_move == PLAYER:
			if user_input: # Manual Input
				action = input('Please decide whether to pass or bet ("p" or "b"): ')
			else: # Get a hardcoded trategy
				action = getStrategy(cards[0], 1)
		else:
			action = getStrategy(cards[1])
			if user_input or verbose:
				print("Your opponent has decided to play:", action)

		history += action

		while not terminal(history):
			plays = len(history)
			player = (player_to_move + plays) % 2
			
			if player == PLAYER:
				if user_input:
					action = input('Please decide whether to pass or bet ("p" or "b"): ')
				else:
					action = getStrategy(cards[0], 1)
			else:
				action = getStrategy(cards[1])
				if user_input or verbose:
					print("Your opponent has decided to play:", action)
			
			history += action

		# Return payoff for terminal states
		terminalPass = history[-1] == 'p'
		doubleBet = history[-2:] == "bb"
		isPlayerCardHigher = cards[0] > cards[1]
		
		
		temp_score = [0, 0]
		if terminalPass:
			if history == "pp": 
				if isPlayerCardHigher:
					temp_score[0] += 1
					temp_score[1] -= 1
				
				else:
					temp_score[0] -= 1
					temp_score[1] += 1
			else: # Equivalent to folding
				temp_score[(first_player_to_move + len(history)) % 2] += 1
				temp_score[(first_player_to_move + len(history) + 1) % 2] -= 1
	
		elif doubleBet:
			if isPlayerCardHigher:
				temp_score[0] += 2
				temp_score[1] -= 2
			
			else: 
				temp_score[0] -= 2
				temp_score[1] += 2
		
		if user_input or verbose:
			if temp_score[0] > temp_score[1]:
				print("Congratulations, you won the round with {} extra chips!\n".format(temp_score[0]))
			else:
				print("You lost to a {} :( You lose {} chips.\n".format(cards[1], temp_score[1]))
			
		score[0] += temp_score[0]
		score[1] += temp_score[1]
		
		# Score scores so it can be plotted afterwards
		user_scores_over_time.append(score[0])
		opponent_scores_over_time.append(score[1])

	plt.plot(user_scores_over_time)
	plt.plot(opponent_scores_over_time)
	if user_input:
		plt.legend(['User Strategy', "CFR Strategy"], loc="upper left")
	else:
		plt.legend(['Deterministic Strategy', "CFR Strategy"], loc="upper left")
	plt.xlabel("Number of Rounds")
	plt.ylabel("Number of Chips Gained")
	plt.savefig("AI_score_over_time.png", bbox_inches='tight')
	plt.show()
		

	
	
