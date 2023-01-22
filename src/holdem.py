
import base
from base import Player, Action
import random
from typing import NewType, Dict, List, Callable, cast
import copy
from fast_evaluator import Deck
from abstraction import *


class HoldEmHistory(base.History):
	"""
	Example of history: 
	First two actions are the cards dealt to the players. The rest of the actions are the actions taken by the players.
		1. ['AkTh', 'QdKd', 'b2', 'c', '/', 'QhJdKs', 'b2', 'c', '/', 'k', 'k']
	
	Actions
	k = check
	bX = bet X amount (this includes raising)
	c = call
	f = fold (you cannot fold if the other player just checked)
	
	Every round starts the same way: 
	Small blind = 1 chip
	Big blind = 2 chips
	
	Total chips = 100BB per player. 
	Minimum raise = X to match bet, and Y is the raise amount
	If no raise before, then the minimum raise amount is 2x the bet amount (preflop would be 2x big blind).
	Else it is whatever was previously raised. This is not the same as 2x the previous bet amount. Just the Y raise amount.
	
	Ex: The bet is 10$. I raise to 50$, so I raised by 40$ (Y = 40). The next player's minimum raise is not 100$, but rather to 90$, since (it's 50$ to match the bet, and 40$ to match the raise).

	Minimum bet = 1 chip (0.5BB)
	
	The API for the history is inspired from the Slumbot API.
	
	I want to avoid all the extra overhead, so taking inspiration from `environment.py` with the `PokerEnvironment`


	"""
	def __init__(self, history: List[Action] = []):
		super().__init__(history)
		
		# How is this going to be taken account of, when we do training?
		
		# # The following values 
		# self.player_0_balance = None
		# self.player_1_balance = None
		# self.total_pot_balance = None
		# self.stage_pot_balance = None
	
	def is_terminal(self):
		if len(self.history) == 0: return False
		folded = self.history[-1] == 'f'
		is_showdown = self.history.count('/') == 3 and self.history[-1] == 'c' # Showdown, since one of the players is calling
		if folded or is_showdown:
			return True
		else:
			return False
	
	def get_current_cards(self):
		current_cards = []
		new_stage = False
		stage_i = 0
		for i, action in enumerate(self.history):
			if new_stage:
				new_stage = False
				if stage_i == 1: # Flop, so there are 3 community cards
					assert(len(action) == 6)
					current_cards.append(action[:2]) # Community card 1
					current_cards.append(action[2:4]) # Community card 2
					current_cards.append(action[4:6]) # Community card 3

				else: # Turn or river
					current_cards.append(action) # Community card
			elif action == '/':
				new_stage = True
				stage_i += 1
			
			elif i == 0 or i == 1:
				assert(len(action) == 4)
				current_cards.append(action[:2]) # Private card 1
				current_cards.append(action[2:4]) # Private card 2

		return current_cards

	def get_current_game_stage_history(self):
		"""
		return current_game_stage_history, stages[stage_i] excluding the community cards drawn. We only care about the actions 
		of the players.
		"""
		game_stage_start = 2 # Because we are skipping the pairs of private cards drawn at the beginning of the round
		stage_i = 0
		stages = ['preflop', 'flop', 'turn', 'river']
		for i, action in enumerate(self.history):
			if action == '/':
				game_stage_start = i + 2 # Skip the community card
				stage_i += 1

		if game_stage_start >= len(self.history):
			return [], stages[stage_i]
		else:
			current_game_stage_history = self.history[game_stage_start:]
			return current_game_stage_history, stages[stage_i]

	def actions(self): # TODO: This is really important to get it right
		if self.is_chance(): # Time to draw cards
			if len(self.history) > 2 and self.history[-1] != '/':
				return ['/'] # One must move onto a new game stage, this is what they do in slumbot as well
			else:
				cards_to_exclude = self.get_current_cards()
				cards = Deck(cards_to_exclude)
				return cards

		elif not self.is_terminal():
			assert(not self.game_stage_ended()) # game_stage_ended would mean that it is a chance node
			"""
			To limit this game going to infinity, I only allow for 3 betting rounds.
			I.e. if I bet, you raise, I raise, you raise, then I must either call, fold, or all-in. Else the branching factor is going to be insane.
			"""

			actions = ['k', 'c', 'f'] 
			player = self.player() 
			remaining_amount = self.get_remaining_balance(player)
			min_bet = self.get_min_bet()

			for bet_size in range(min_bet, remaining_amount + 1): # These define the legal actions of the game
				actions.append('b' + str(bet_size))

			current_game_stage_history, stage = self.get_current_game_stage_history()
			# Pre-flop
			if stage == 'preflop':
				# Small blind to act
				if len(current_game_stage_history) == 0: # Action on SB (Dealer), who can either call, bet, or fold
					actions.remove('k') # You cannot check
					return actions

				# big blind to act
				elif len(current_game_stage_history) == 1: # 2-bet
					if (current_game_stage_history[0] == 'c'): # Small blind called, you don't need to fold
						actions.remove('f')
						return actions
					else: # Other player has bet, so you cannot check
						actions.remove('k')
						return actions
				else:
					actions.remove('k')

				# elif len(current_game_stage_history) == 2: # 3-bet
				# 	# You cannot check at this point
				# 	actions = ['b1', 'all-in', 'c', 'f']
					
				# elif len(current_game_stage_history) == 3: # 4-bet
				# 	actions = ['all-in', 'c', 'f'] 

			else: # flop, turn, river
				if len(current_game_stage_history) == 0:
					actions.remove('f') # You cannot fold
				elif len(current_game_stage_history) == 1:
					if current_game_stage_history[0] == 'k':
						actions.remove('f')
					else: # Opponent has bet, so you cannot check
						actions.remove('k')
				else:
					actions.remove('k') # Cannot check

			return actions
		else:
			raise Exception("Cannot call actions on a terminal history")

	def get_min_bet(self):
		# TODO: Test this function
		curr_bet = 0
		prev_bet = 0
		for i in range(len(self.history)-1, 0, -1):
			if self.history[i][0] == 'b': # Bet, might be a raise
				if curr_bet == 0:
					curr_bet = int(self.history[i][1:])
				elif prev_bet == 0:
					prev_bet = int(self.history[i][1:])
			elif self.history[i] == '/':
				break

		# Handle case when game stage is preflop, in which case a bet is already placed for you
		game_stage_history, game_stage = self.get_current_game_stage_history()
		if game_stage == 'preflop' and curr_bet == 0:
			curr_bet = 2 # big blind
		elif curr_bet == 0: # No bets has been placed
			assert(prev_bet == 0)
			curr_bet = 1

		return int(curr_bet + (curr_bet - prev_bet)) # This is the minimum raise


	def calculate_player_total_up_to_game_stage(self, player: Player):
		stage_i = 0
		player_total = 0 # Total across all game stages (preflop, flop, turn, river)
		player_game_stage_total = 0 # Total for a given game stage
		i = 0
		for hist_idx, hist in enumerate(self.history):
			i = (i + 1) % 2
			if i == player:
				if hist[0] == 'b':
					player_game_stage_total = int(hist[1:])
				elif hist == 'k':
					if stage_i == 0: # preflop, checking means 2
						player_game_stage_total = 2
					else:
						player_game_stage_total = 0
				elif hist == 'c': # Call the previous bet
					# Exception for when you can call the big blind on the preflop, without the big blind having bet previously
					if hist_idx == 2:
						player_game_stage_total = 2
					else:
						player_game_stage_total = int(self.history[hist_idx - 1][1:])

			if hist == '/':
				stage_i += 1
				player_total += player_game_stage_total
				player_game_stage_total = 0
				if stage_i == 1:
					i = (i + 1) % 2 # We need to flip the order post-flop, as the BB is the one who acts first now

		return player_total
		

	def get_remaining_balance(self, player: Player):
		"""
		Calculate the remaining balance for the given player
		
		Each player has a balance of 100
		"""
		return 100 - self.calculate_player_total_up_to_game_stage(player)

	def game_stage_ended(self):
		# TODO: Make sure this logic is good
		current_game_stage_history, stage = self.get_current_game_stage_history()
		if len(current_game_stage_history) == 0:
			return False
		elif current_game_stage_history[-1] == 'f':
			return True
		elif current_game_stage_history[-1] == 'c' and len(self.history) > 3: # On pre-flop, when the small blind calls, the opponent can still bet
			return True
		elif len(current_game_stage_history) >= 2 and current_game_stage_history[-2:] == ['k', 'k']:
			return True
		else:
			return False

	def player(self):
		# TODO: check that this is correct
		"""
		This part is confusing for heads-up no limit poker, because the player that acts first changes:
		The Small Blind (SB) acts first pre-flop, but the Big Blind (BB) acts first post-flop.
		1. ['AkTh', 'QdKd', 'b2', 'c', '/', 'Qh', 'b2', 'c', '/', '2d', b2', 'f']
							 SB	   BB		 	   BB	 SB 	   		BB 	 SB
		"""
		# Chance nodes, we have to draw cards
		if len(self.history) <= 1: 
			return -1
		
		# End of a game stage, we need to draw a new card
		elif self.game_stage_ended():
			# Next action is '/'
			return -1
		elif self.history[-1] == '/':
			# Next action is drawing one of the community cards
			return -1
		else:
			if '/' in self.history:
				return (len(self.history) + 1) % 2 # Order is flipped post-flop
			else:
				return len(self.history) % 2 
	
	def is_chance(self):
		return super().is_chance()
	
	def sample_chance_outcome(self):
		assert(self.is_chance())

		cards = self.actions() # Will be either or cards not seen in the deck or ['/']
		
		if len(self.history) <= 1: # We need to deal two cards to each player
			cards = random.sample(cards, 2)
			return ''.join(cards)
		else:
			return random.choice(cards) # Sample one of the community cards with equal probability


	def terminal_utility(self, i: Player) -> int:
		# TODO: Check if this is accurate
		assert(self.is_terminal()) # We can only call the utility for a terminal history
		assert(i in [0, 1]) # Only works for 2 player games for now
		
		actions = ['k', 'b1', 'b2', 'b4', 'b8', 'all-in', 'c', 'f'] 
		pot_size = 0
		# These represent the bets in the current game stage, i.e. pre-flop, flop, turn, river
		prev_bet = 1 # small blind starting value
		curr_bet = 2 # big blind starting value
		for i, action in enumerate(self.history):
			if action == '/': # Move on to next stage
				assert(curr_bet == prev_bet and curr_bet == 0)
				pot_size += curr_bet
				prev_bet = 0

			if action not in actions:
				continue

			if action == 'k':
				assert(curr_bet == prev_bet and curr_bet == 0)

			elif action == 'b1':
				assert(curr_bet == 0)
				curr_bet = 1

			elif action == 'b2':
				if curr_bet == 0:
					assert(prev_bet == 0)
					curr_bet = 2
				else:
					prev_bet = curr_bet
					curr_bet *= 2
			elif action == 'b4':
				if curr_bet == 0:
					assert(prev_bet == 0)
					curr_bet = 4
				else:
					prev_bet = curr_bet
					curr_bet *= 4
			elif action == 'b8':
				if curr_bet == 0:
					assert(prev_bet == 0)
					curr_bet = 8
				else:
					prev_bet == curr_bet
					curr_bet *= 8
			
			elif action == 'all-in':
				curr_bet = 100 - pot_size - curr_bet # Maximum, since each player has 100 chips

			elif action == 'c':
				assert(curr_bet != 0)
				pot_size += 2 * curr_bet
				curr_bet = 0
				prev_bet = 0

			elif action == 'f':
				assert(i == len(self.history) - 1) # Folding should be the last action

				pot_size += prev_bet
				pot_size += curr_bet
			
			else:
				raise Exception("Action not recognized")

		# Now that we know how much we won from the pot, we also we to calculate how much we made ourselves
				
	
	def __add__(self, action: Action):
		new_history = HoldEmHistory(self.history + [action])
		return new_history
	
	def get_infoSet_key(self, kmeans_flop = None, kmeans_turn = None, kmeans_river = None) -> List[Action]:
		"""
		This gets a little complicated with card abstraction.
		
		I implement imperfect-recall abstraction, which means that you lose some information. This means that at each layer, 
		there is a different level of abstraction.
		
		Preflop
		2 private cards -> Hash bucket
		
		Flop
		2 private cards + 3 community cards -> Hash bucket
		
		"""
		

		assert(not self.is_chance()) # chance history should not be infosets
		assert(not self.is_terminal())

		player = self.player()
		
		"""
		There are so many decision points.
		
		As a starting part, assume that all the cards are the same, and just run CFR for a particular bet sequence.
		"""

		if player == 0:
			history = copy.deepcopy(self.history)
			history[1] = '?' # Unknown card
		else:
			history = copy.deepcopy(self.history)
			history[0] = '?' # Unknown card
			
		# TODO: Replace with proper card abstraction, right now replaces all of them with '?'
		if kmeans_flop is not None:
			"""
			['preflop4', '?', 'b2', 'c', '/', 'flop1', 'b2', 'c', '/', 'k', 'k']
			
			"""

			cards = self.get_current_cards()
			print("Cards running under get_infoSet_key(): ", cards)
			if player == 0:
				if len(cards) >= 2:
					cluster_id = "preflop" + str(get_preflop_cluster_id(cards[0:2]))
					history[0] = cluster_id

				if len(cards) == 5:
					cluster_id = "flop" + str(get_flop_cluster_id(kmeans_flop, cards))
				if len(cards) == 6:
					cluster_id = "turn" + str(get_turn_cluster_id(kmeans_turn, cards))
				if len(cards) == 7:
					cluster_id = "river" + str(get_river_cluster_id(kmeans_river, cards))
			else:
				raise Exception("Invalid number of cards")

		return history


class HoldemInfoSet(base.InfoSet):
	"""
	Information Sets (InfoSets) cannot be chance histories, nor terminal histories.
	This condition is checked when infosets are created.
	
	This infoset is an abstracted versions of the history in this case. 
	See the `get_infoSet_key(self)` function for these
	
	There are 2 abstractions we are doing:
		1. Card Abstraction (grouping together similar hands)
		2. Action Abstraction
	
	I've imported my abstractions from `abstraction.py`.
	
	"""
	def __init__(self, infoSet: List[Action], actions: List[Action], player: Player):
		assert(len(infoSet) >= 2)
		super().__init__(infoSet, actions, player)
		

def create_infoSet(infoSet_key: List[Action], actions: List[Action], player: Player):
	"""
	We create an information set from a history. 
	"""
	return HoldemInfoSet(infoSet_key, actions, player)
	
	
def create_history():
	return HoldEmHistory()
	
	
# CFR with abstraction integrated
class AbstractCFR(base.CFR):
	def __init__(self, create_infoSet, create_history, n_players: int = 2, iterations: int = 1000000):
		super().__init__(create_infoSet, create_history, n_players, iterations)
	

if __name__ == "__main__":
	kmeans_flop, kmeans_turn, kmeans_river = load_kmeans_classifiers()
	cfr = base.CFR(create_infoSet, create_history)
	cfr.solve()
	
	

	

	# hist: HoldEmHistory = create_history()
	# assert(hist.player() == -1)
	# hist1 = hist + 'AkTh'
	# assert(hist1.player() == -1)
	# hist2 = hist1 + 'QdKd'
	# assert(hist2.player() == 0)
	# hist3 = hist2 + 'b2'
	# assert(hist3.player() == 1)
	# hist4 = hist3 + 'c'
	# assert(hist4.player() == -1)
	# # Below are chance events, so it doesn't matter which player it is
	# hist5 = hist4 + '/'
	# assert(hist5.player() == -1)
	# hist6 = hist5 + 'QhKsKd'
	# assert(hist6.player() == 1)
	# hist7 = hist6 + 'b1'
	# hist8: HoldEmHistory = hist7 + 'b3'
	# print(hist8.get_infoSet_key())

	# cfr = base.CFR(create_infoSet, create_history)
	# cfr.solve()