
"""
Before looking at this code, you should probably familiarize with a simpler implementation
with Kuhn Poker, under `research/kuhn/train.py`

This is the main training file for Poker CFR.
"""
from typing import List
from evaluator import Deck, CombinedHand, Evaluator, Card
import numpy as np
import joblib
import copy
import argparse
from tqdm import tqdm

"""
Starting off with simple training, we have the following 3 actions:
1) Fold
2) Check/Call
3) Bet/Raise
"""
NUM_ACTIONS = 3 
nodeMap = {}

class Node:
	def __init__(self) -> None:
		self.infoSet = ""
		self.regretSum = np.zeros(NUM_ACTIONS)
		self.strategy = np.zeros(NUM_ACTIONS)
		self.strategySum = np.zeros(NUM_ACTIONS)
	
	def describe(self):
		print("Infoset: {} -> Strategy at this infoset: {}, RegretSum: {}".format(self.infoSet, np.around(self.getAverageStrategy(), 2), self.regretSum.sum()))

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


class History():
	# SB bet size = 1, BB bet size = 2
	def __init__(self):
		self.total_pot_size = 0
		self.history_str = ""
		self.min_bet_size = 2
		self.game_stage = 2
		self.curr_round_plays = 0 # if self.curr_round_plays == 0 and we check, then we DON'T move to the next game stage

	

all_history = [] # Global variable to store all histories
def cfr(all_community_cards: List[Card], private_cards: List[CombinedHand], community_cards: CombinedHand,  history: History, p0, p1):
	"""
	player_cards: [user_cards, opponent_cards]
	community_cards: "" for community (board) cards
	
	To compare cards, we get the binary representation.
	"""
	print(p0, p1)
	
	# print(history.history_str)
	plays = len(history.history_str)
	player = plays % 2
	opponent = 1 - player
	
	# Return payoff for terminal states
	if (plays >= 1):
		if history.history_str[-1] == 'f': # Fold, just calculate total value
			all_history.append(history.history_str)
			return history.total_pot_size
		
		elif history.game_stage == 6:
			# Showdown
			all_history.append(history.history_str)
			hand = copy.deepcopy(CombinedHand())
			hand.add_combined_hands(community_cards, private_cards[player])

			opponent_hand = copy.deepcopy(CombinedHand())
			opponent_hand.add_combined_hands(community_cards, private_cards[opponent])

			evaluator = copy.deepcopy(Evaluator())
			evaluator.add_hands(hand, opponent_hand)
			
			assert(len(hand) == 7)
			assert(len(opponent_hand) ==  7)
			assert(len(evaluator.hands) == 2)

			winners = evaluator.get_winner()
			# print("Showdown time! Winner(s):", winners)
			
			# if (len(winners) == 0):
			# 	print("Curent Hand", len(hand), str(hand))
			# 	print("Opponent Hand", str(opponent_hand))
			# 	print("Community Cards", str(community_cards))
				# print(str(community_cards))

			assert(len(winners) > 0) # At least one winner

			if len(winners) == 2: # Tie
				return history.total_pot_size / 2
			else:
				if winners[0] == 0:
					return history.total_pot_size
				else:
					return - history.total_pot_size
	
	if community_cards == None:
		infoSet = private_cards[player].get_binary_representation() + history.history_str
	else:
		infoSet = private_cards[player].get_binary_representation() + community_cards.get_binary_representation() + history.history_str
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
		nextHistory = copy.deepcopy(history)
		new_community_cards = copy.deepcopy(community_cards)
		
		nextHistory.curr_round_plays += 1

		if a == 0:
			nextHistory.history_str += "f" # fold

		elif a == 1:
			nextHistory.history_str += "c" # Check/Call
			nextHistory.total_pot_size += nextHistory.min_bet_size
			if nextHistory.curr_round_plays > 1: # We move to to the next game_stage if there is more than one play
				nextHistory.game_stage += 1
				nextHistory.curr_round_plays = 0
				nextHistory.min_bet_size = 0 # You don't have to bet anything

				if nextHistory.game_stage == 3: # Flop
					new_community_cards = CombinedHand(all_community_cards[:3])
					assert(len(new_community_cards) == 3)
				elif nextHistory.game_stage == 4: # Turn
					new_community_cards.add_cards(all_community_cards[3])
					assert(len(new_community_cards) == 4)
				elif nextHistory.game_stage == 5: # River
					new_community_cards.add_cards(all_community_cards[4])
					assert(len(new_community_cards) == 5)
				
		else:
			nextHistory.history_str += "r" # Bet/Raise
			if (len(nextHistory.history_str) > 3) and nextHistory.history_str[-3:] == 'rrr': continue # To prevent infinite raises, we just don't consider this node
			
			if nextHistory.min_bet_size == 0:
				nextHistory.min_bet_size = 1
			else:
				nextHistory.min_bet_size *= 2
			
			nextHistory.total_pot_size += nextHistory.min_bet_size

		
		util[a] = - cfr(all_community_cards, private_cards, new_community_cards, nextHistory, p0 * strategy[a], p1) if player == 0 else - cfr(all_community_cards, private_cards, new_community_cards, nextHistory, p0, p1 * strategy[a])
		nodeUtil += strategy[a] * util[a]
	
	# For each action, compute and accumulate counterfactual regret
	for a in range(NUM_ACTIONS):
		regret = util[a] - nodeUtil
		node.regretSum[a] += (p1 if player == 0 else p0) * regret
	return nodeUtil

def train(iterations):
	deck = Deck()
	util = 0
	for i in tqdm(range(iterations), desc="Training Loop"):
		deck.reset_deck()
		player_cards = CombinedHand([deck.draw(), deck.draw()])
		opponent_cards = CombinedHand([deck.draw(), deck.draw()])
		
		all_community_cards = []
		for _ in range(5):
			all_community_cards.append(deck.draw())
		
		assert(deck.total_remaining_cards == 43)
		
		private_cards = [player_cards, opponent_cards]
		community_cards = None
		history = History()
		util += cfr(all_community_cards,private_cards,community_cards,history,1,1)
		if (i % 1 == 0):
			print("Average game value: ", util/i)
	
	
if __name__ == "__main__":
	train_from_scratch = True # Set this to True if you want to retrain from scratch
	parser = argparse.ArgumentParser(description="Train a Hold'Em AI.")
	parser.add_argument("-s", "--save",
                    action="store_true", dest="save", default=True,
                    help="Save the trained model and history")
	parser.add_argument("-v", "--visualize",
                    action="store_true", dest="visualize", default=False,
                    help="Print out all information sets with their corresponding strategy.")

	args = parser.parse_args()
	save = args.save
	visualize = args.visualize
	if not visualize:
		train(10)
		if save:
			joblib.dump(nodeMap, "HoldemNodeMap.joblib")
			joblib.dump(all_history, "HoldemTrainingHistory.joblib")
	else:
		nodeMap = joblib.load("HoldemNodeMap.joblib")

	print("Total Number of Infosets:", len(nodeMap))
	for infoset in nodeMap:
		nodeMap[infoset].describe()