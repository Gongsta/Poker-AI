"""
This is a fast evaluator used for training. It works with string representation of
cards. However, it cannot tell you if you won with a pair, three of a kind, etc.
"""
import random 
from phevaluator import evaluate_cards

def Deck(excluded_cards=[]):
	# Returns a shuffled deck
	deck = []
	for rank in ["A", "2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K"]:
		for suit in ["h", "d", "s", "c"]:
			if (rank+suit not in excluded_cards):
				deck.append(rank + suit)
    
	random.shuffle(deck)
	return deck

def get_player_score(player_cards, board=[]):
	"""Wrapper for the evaluate_cards function by phevaluator.
	
	"""
	assert(len(player_cards) == 2)
	assert(len(board) <= 5)
	# Returns a score using the phevaluator library
	return evaluate_cards(*(player_cards + board))