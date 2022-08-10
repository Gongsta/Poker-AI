# The Poker Environment
import random
from typing import List

CARD_RANKS = [i for i in range(2, 15)] # Jack = 11, Queen = 12, King = 13, IMPORTANT: Ace = 14 since we use that for sorting
CARD_SUITS = ["Clubs", "Diamonds", "Hearts","Spades"] 

RANK_KEY = {"A": 14, "2": 2, "3": 3, "4":4, "5":5, "6":6,
			"7": 7, "8": 8, "9": 9, "10": 10, "J": 11, "Q": 12, "K":13}

INVERSE_RANK_KEY = {14: "A", 2: "02", 3: "03", 4:"04", 5:"05", 6:"06",
			7:"07", 8:"08", 9:"09", 10:"10", 11: "J", 12: "Q", 13: "K"}

SUIT_KEY = {"C": "Clubs", "D": "Diamonds", "H":"Hearts","S": "Spades"}

class Card():
	# Immutable after it has been initialized
	def __init__(self, rank=1, suit="Spades", rank_suit=None, generate_random=False) -> None:

		if rank_suit: # Ex: "KD" (King of diamonds), "10H" (10 of Hearts),
			self.__suit = SUIT_KEY[rank_suit[-1]]
			self.__rank = RANK_KEY[rank_suit[:-1]]

		else:
			self.__rank = rank
			self.__suit = suit
		
		if generate_random: # If we want to just generate a random card
			self.__rank = random.choice(CARD_RANKS)
			self.__suit = random.choice(CARD_SUITS)
	
		# Check validity of card TODO: Maybe put into separate function to check wellformedness
		if self.__rank not in CARD_RANKS:
			raise Exception("Invalid Rank: {}".format(self.__rank))
		if self.__suit not in CARD_SUITS: 
			raise Exception("Invalid Suit: {}".format(self.__suit))

	@property
	def rank(self):
		return self.__rank

	@property
	def suit(self):
		return self.__suit
	
	def print(self):
		print("  ", self.rank, "of", self.suit)
	

class Deck():
	def __init__(self) -> None: # Create a new full deck
		self.__cards: List[Card] = []
		self.reset_deck()
	
	def shuffle(self):
		random.shuffle(self.__cards)

	def reset_deck(self):
		self.__cards = []
		for rank in CARD_RANKS:
			for suit in CARD_SUITS:
				self.__cards.append(Card(rank, suit))
		
		random.shuffle(self.__cards)


	@property
	def total_remaining_cards(self):
		return len(self.__cards)

	def draw(self): # Draw a card from the current deck
		card = self.__cards.pop()
		return card
		
ACTIONS = ["Check", "Bet", "Call", "Fold"]

class Player(): # This is the POV
	def __init__(self, balance) -> None:
		self.is_AI = False

		self.hand: List[Card] = [] # The hand is also known as hole cards: https://en.wikipedia.org/wiki/Texas_hold_%27em
		self.player_balance = balance # TODO: Important that this value cannot be modified easily...
		self.current_bet = 0

		
		self.playing_current_round = True
		

	# Wellformedness, hand is always either 0 or 2 cards
	
	def add_card_to_hand(self, card: Card):
		self.hand.append(card)
		assert(len(self.hand) <= 2)
	
	def clear_hand(self):
		self.hand = []
	
	def set_current_bet(self, bet: int):
		self.current_bet = bet

	# def update_observed_environment(self, env: PokerEnvironment): # Partially observed environment
	# 	self.observed_env = env


	def place_bet(self, action: str, observed_env) -> int:
		if action == "Check":
			if observed_env.min_bet_size > 0: 
				return Exception("You cannot check, since there is a bet. You can either 'Bet' or 'Fold'.")

			else:
				self.set_current_bet(0)

		# TODO: This logic is bad
		elif action == "Bet":
			self.set_current_bet(observed_env.small_blind)
		
		elif action ==  "Call": # 
			self.set_current_bet(observed_env.big_blind)

		elif action == "Fold":
			self.playing_current_round = False # Balance of all players will be updated at the end of the round
		else:
			raise Exception("Invalid Action")
		
		return self.current_bet



class AIPlayer(Player):
	def __init__(self, balance) -> None:
		super().__init__(balance)
		self.is_AI = True
	
	# We are going to have the dumbest AI possible, which is to 
	def place_bet(self, observed_env) -> int: # AI will bet every single round
		self.set_current_bet(observed_env.min_bet_size)
		
		return self.current_bet

class PokerEnvironment():
	def __init__(self) -> None:
		self.players: List[Player] = []
		self.deck = Deck() 
		
		"""Game States:
		1: Starting a new round, giving players their cards. Automatically goes into state 2
		2: Preflop betting round. Goes into state 3 once everyone has made their decision
		3: Flop round. Goes into turn (state 4) /ends round (state 6) once everyone " " 
		4: Turn round. Goes into river (state 5) /ends round (state 6) once everyone " " 
		5: River round. Ends round (state 6) once everyone " " 
		6: Round is over. Distribute pot winnings.
		"""
		self.game_stage = 1 # To keep track of which phase of the game we are at, new_round is 0
# If self.finished_playing_game_stage = True, we can move to the next game state. This is needed to go around each player and await their decision
		self.finished_playing_game_stage = False 

		# Changes every round
		self.dealer_button_position = 0 # This button will move every round
		self.total_pot_balance = 0 # keep track of pot size of total round
		self.stage_pot_balance = 0 # keep track of pot size for current round
		self.community_cards: List[Card] = [] # a.k.a. the board
		self.min_bet_size = 0
		self.position_in_play = 0
		
		self.first_player_to_place_highest_bet = 0 # This is to keep track of who is the first player to have placed the highest bet, so we know when to end the round

		
		# These values should rarely change
		self.new_player_balance = 10000
		self.small_blind = 50 
		self.big_blind = 100 # TODO: Check if this is even necessary 
		
	
	def add_player(self):
		self.players.append(Player(self.new_player_balance))

	def get_player(self, idx) -> Player:
		return self.players[idx]

	def add_AI_player(self): # Add a dumb AI
		self.players.append(AIPlayer(self.new_player_balance))
	
	def get_winning_players(self) -> List:
		# If there is more than one winning player, the pot is split. We assume that we only run things once
		winning_players: List = []
		for player in self.players:
			if player.playing_current_round:
				winning_players.append(player)

		return winning_players

	def distribute_pot_to_winning_players(self): # Run when self.game_stage = 5
		winning_players = self.get_winning_players()
		
		pot_winning = self.total_pot_balance / len(winning_players)
		for player in winning_players:
			player.player_balance += pot_winning

		self.total_pot_balance = 0 # Reset the pot just to be safe
		self.stage_pot_balance = 0 # Reset the pot just to be safe

	def count_remaining_players_in_round(self):
		# Helper function to count the total number of players still in the round
		total = 0
		for player in self.players:
			if player.playing_current_round:
				total += 1
		return total
	
	def print_board(self):
		for card in self.community_cards:
			card.print()
		
	def start_new_round(self):
		assert(len(self.players) >= 2) # We cannot start a poker round with less than 2 players...
		
		# Reset Players
		for player in self.players: 
			player.playing_current_round = True
			player.set_current_bet(0)
			player.clear_hand()

		# Reset Deck (shuffles it as well), reset pot size 
		self.deck.reset_deck()
		self.community_cards = []
		self.stage_pot_balance = 0
		self.total_pot_balance = 0

		# Move the dealer position and assign the new small and big blinds
		self.dealer_button_position += 1
		self.dealer_button_position %= len(self.players)
		
		# Small Blind
		self.players[((self.dealer_button_position + 1) % len(self.players))].set_current_bet(self.small_blind)

		# Big Blind
		self.players[((self.dealer_button_position + 2) % len(self.players))].set_current_bet(self.big_blind)
		
		self.update_stage_pot_balance()
		# 3. Deal Cards
		# We start dealing with the player directly clockwise of the dealer button
		position_to_deal = self.dealer_button_position + 1 

		for _ in range(2):# Deal the cards to each player, moving in a clockwise circle twice
			for _ in range(len(self.players)): 
				position_to_deal %= len(self.players)

				card = self.deck.draw()
				self.players[position_to_deal].add_card_to_hand(card)
				
				position_to_deal += 1

		self.finished_playing_game_stage = True

	def update_stage_pot_balance(self):
		self.stage_pot_balance = 0
		for player in self.players:
			self.stage_pot_balance += player.current_bet

	def play_current_stage(self, action: str = ""):
		self.update_stage_pot_balance()
		if self.players[self.position_in_play].is_AI:
			player_bet = self.players[self.position_in_play].place_bet(self) # Pass the Environment as an argument
			
		else: # Real player's turn
			if action == "": # No decision has yet been made
				return 
			else:
				player_bet = self.players[self.position_in_play].place_bet(action, self)

		self.min_bet_size = max(player_bet, self.min_bet_size)
		self.update_stage_pot_balance()
			
		print(self.count_remaining_players_in_round())
		if self.count_remaining_players_in_round() == 1: # Round is over, distribute winnings
			self.finished_playing_game_stage = True
			self.game_stage = 6
			return 
		else:
			self.move_to_next_player()
			
		if self.position_in_play == self.first_player_to_place_highest_bet: # Stage is over, move to the next stage (see flop)
			self.finished_playing_game_stage = True

	def move_to_next_player(self):
		assert(self.count_remaining_players_in_round() > 1)
		self.position_in_play += 1
		self.position_in_play %= len(self.players)

		while (not self.players[self.position_in_play].playing_current_round):
			self.position_in_play += 1
			self.position_in_play %= len(self.players)
		
	def play_preflop(self):
		self.position_in_play = self.dealer_button_position + 3
		self.position_in_play %= len(self.players)

		self.min_bet_size = self.big_blind
		
		self.finished_playing_game_stage = False

	def play_flop(self):
		# 3. Flop
		self.min_bet_size = 0
		self.deck.draw() # We must first burn one card, TODO: Show on video
		for _ in range(3): # Draw 3 cards
			self.community_cards.append(self.deck.draw())

		self.finished_playing_game_stage = False
				
	def play_turn(self):
		self.min_bet_size = 0 
		# 4. Turn
		self.deck.draw() # We must first burn one card, TODO: Show on video
		self.community_cards.append(self.deck.draw())
		
		self.finished_playing_game_stage = False
			
	def play_river(self):
		self.min_bet_size = 0
		# 5. River
		self.deck.draw() # We must first burn one card, TODO: Show on video
		self.community_cards.append(self.deck.draw())

		self.finished_playing_game_stage = False
		
		
	def update_player_balances_at_end_of_stage(self):
		for player in self.players:
			player.player_balance -= player.current_bet
			player.current_bet = 0
		
	def move_stage_to_total_pot_balance(self):
		self.total_pot_balance += self.stage_pot_balance
		self.stage_pot_balance = 0

	def handle_game_stage(self, action=""):
		# print(self.game_stage, self.finished_playing_game_stage)
		if self.finished_playing_game_stage:
			if self.game_stage != 1:
				self.update_player_balances_at_end_of_stage()
				self.move_stage_to_total_pot_balance()
			self.game_stage += 1
			
			if self.game_stage == 2:
				self.play_preflop()
			elif self.game_stage == 3:
				self.play_flop()
			elif self.game_stage == 4:
				self.play_turn()
			elif self.game_stage == 5:
				self.play_river()
			else:
				self.distribute_pot_to_winning_players()
				self.game_stage = 1
				self.finished_playing_game_stage = False # on the next call of the handler, we will start a new round
		else:
			if self.game_stage == 1:
				self.start_new_round()
			else:
				self.play_current_stage(action)
		