
import base
from base import Player, Action
import random
from typing import NewType, Dict, List, Callable, cast
import copy
from fast_evaluator import Deck

deck = Deck()


class HoldEmHistory(base.History):
	"""
	Example of history: 
	First two actions are the cards dealt to the players. The rest of the actions are the actions taken by the players.
		1. ['AkTh', 'QdKd', 'b2', 'c', '/', 'Qh', 'b2', 'c', '/', 'k', 'k']
	
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

	"""
	def __init__(self, history: List[Action] = []):
		super().__init__(history)
		self.player_
	
	def is_terminal(self):
		folded = self.history[-1] == 'f'
		is_showdown = self.history.count('/') == 3 and self.history[-1] == 'c' # Showdown, since one of the players is calling
		if folded or is_showdown:
			return True
		else:
			return False
	
	def get_current_cards(self):
		current_cards = []
		new_stage = False
		for i, action in enumerate(self.history):
			if new_stage:
				new_stage = False
				current_cards.append(action) # Community card
			elif action == '/':
				new_stage = True
			
			elif i == 0 or i == 1:
				assert(len(action) == 4)
				current_cards.append(action[:2]) # Private card 1
				current_cards.append(action[2:4]) # Private card 2

	def get_current_game_stage_history(self):
		game_stage_start = 0
		stage_i = 0
		stages = ['preflop', 'flop', 'turn', 'river']
		for i, action in enumerate(self.history):
			if action == '/':
				game_stage_start = i + 2 # Skip the community card
				stage_i += 1

		if game_stage_start >= len(self.history) or game_stage_start == 0:
			return [], stages[stage_i]
		else:
			current_game_stage_history = self.history[game_stage_start:]
			return current_game_stage_history, stages[stage_i]

	def actions(self):
		if self.is_chance(): # Time to draw cards
			if len(self.history) > 2 and self.history[-1] != '/':
				return ['/'] # One must move onto a new game stage, though this is not a very elegant solution? TODO: Improve
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
			# TODO: Investigate the effect of action abstraction on exploitability.

			"""
			
			Daniel Negreanu: How Much Should You Raise? https://www.youtube.com/watch?v=WqRUyYQcc5U
			Bet sizing: https://www.consciouspoker.com/blog/poker-bet-sizing-strategy/#:~:text=We%20recommend%20using%201.5x,t%20deduce%20your%20likely%20holdings.
			Also see slumbot notes: https://nanopdf.com/queue/slumbot-nl-solving-large-games-with-counterfactual_pdf?queue_id=-1&x=1670505293&z=OTkuMjA5LjUyLjEzOA==

			For initial bets, these are fractions of the total pot size (money at the center of the table):
			for bets:
				- b0.25 = bet 25% of the pot 
				- b0.5 = bet 50% of the pot
				- b0.75 = bet 75% of the pot
				- b1 = bet 100% of the pot
				- b2 = ...
				- b4 = ...
				- b8 =
				- all-in = all-in, opponent is forced to either call or fold

			After a bet has happened, we can only raise by a certain amount.
				- b0.5
				- b1 
				- b2 = 2x pot size
				- b4 = 4x pot size
				- b8 = 8x pot size
				- all-in = all-in, opponent is forced to either call or fold
			
			2-bet is the last time we can raise again
			- b1
			- b2 = 2x pot size
			- all-in
			
			3-bet
			- b1
			
			4-bet
			- all-in
			
			# TODO: Handle all-in case
			
			It's just so annoying because there are so many bet sizing edge cases to take care off.
			
			"""
			actions = ['k', 'b0.25','b0.5', 'b0.75', 'b1', 'b2', 'b4', 'b8', 'all-in', 'c', 'f'] 
			
			current_game_stage_history, stage = self.get_current_game_stage_history()

			# Pre-flop
			if stage == 'preflop':
			# Small blind to act
				if len(current_game_stage_history) == 0: # call/bet
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
				elif len(current_game_stage_history) == 2: # 3-bet
					# You cannot check at this point
					actions = ['b1', 'all-in', 'c', 'f']
					
				elif len(current_game_stage_history) == 3: # 4-bet
					actions = ['all-in', 'c', 'f'] 

			else: # flop, turn, river
				if len(current_game_stage_history == 0):
					actions.remove('f') # You cannot fold
				elif len(current_game_stage_history) == 1:
					if current_game_stage_history[0] == 'k':
						actions.remove('f')
					else: # Opponent has bet, so you cannot check
						actions.remove('k')

			return actions
		else:
			raise Exception("Cannot call actions on a terminal history")

	
	def game_stage_ended(self):
		current_game_stage_history = self.get_current_game_stage_history()
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
		assert(not self.is_terminal())
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
	
	def get_infoSet_key(self) -> List[Action]:
		assert(not self.is_chance()) # chance history should not be infosets
		assert(not self.is_terminal())

		player = self.player()
		if player == 0:
			history = copy.deepcopy(self.history)
			history[1] = '?' # Unknown card
			return history
		else:
			history = copy.deepcopy(self.history)
			history[0] = '?' # Unknown card
			return history

class HoldemInfoSet(base.InfoSet):
	"""
	Information Sets (InfoSets) cannot be chance histories, nor terminal histories.
	This condition is checked when infosets are created.
	
	"""
	def __init__(self, infoSet: List[Action]):
		assert(len(infoSet) >= 2)
		super().__init__(infoSet)

	def actions(self) -> List[Action]:
		return ['p', 'b']
	
	def player(self) -> Player:
		plays = len(self.infoSet)
		if plays <= 1:
			return -1
		else:
			return plays % 2

def create_infoSet(infoSet_key: List[Action]):
	"""
	We create an information set from a history. 
	"""
	return HoldemInfoSet(infoSet_key)
	
	
def create_history():
	return HoldEmHistory()
	

if __name__ == "__main__":
	cfr = base.CFR(create_infoSet, create_history)
	cfr.solve()