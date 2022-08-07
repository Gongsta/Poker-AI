# The Poker Environment
import random
from typing import List

CARD_RANKS = [i for i in range(2, 15)] # Jack = 11, Queen = 12, King = 13, IMPORTANT: Ace = 14 since we use that for sorting
CARD_SUITS = ["Clubs", "Diamonds", "Hearts","Spades"] 

RANK_KEY = {"A": 14, "2": 2, "3": 3, "4":4, "5":5, "6":6,
			"7": 7, "8": 8, "9": 9, "10": 10, "J": 11, "Q": 12, "K":13}

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
		for rank in CARD_RANKS:
			for suit in CARD_SUITS:
				self.__cards.append(Card(rank, suit))
		
		self.shuffle()
	
	def shuffle(self):
		random.shuffle(self.__cards)

	@property
	def total_remaining_cards(self):
		return len(self.__cards)

	def draw(self): # Draw a card from the current deck
		card = self.__cards.pop()
		return card
		
ACTIONS = ["Check", "Bet", "Call", "Fold"]

class PokerEnvironment():
	def __init__(self) -> None:
		self.players: List[Player] = []
		self.deck = Deck() 
		
		# Changes every round
		self.dealer_button_position = 0 # This button will move every round
		self.pot_balance = 0 # keep track of the current pot size
		self.community_cards: List[Card] = [] # a.k.a. the board
		self.min_bet_size = 0

		
		# These values should rarely change
		self.new_player_balance = 10000
		self.small_blind = 50 
		self.big_blind = 100 # TODO: Check if this is even necessary 
	
	def add_player(self):
		self.players.append(Player(self.new_player_balance))

	def add_AI_player(self): # Add a dumb AI
		self.players.append(AIPlayer(self.new_player_balance))
	
	def get_winning_players(self) -> List:
		# If there is more than one winning player, the pot is split. We assume that we only run things once
		winning_players: List = []
		for player in self.players:
			if player.playing_current_round:
				winning_players.append(player)

		return winning_players

	def distribute_pot_to_winning_players(self):
		winning_players = self.get_winning_players()
		
		pot_winning = self.pot_balance / len(winning_players)
		for player in winning_players:
			player.player_balance += pot_winning

		self.pot_balance = 0 # Reset the pot just to be safe

	def get_remaining_players_in_round(self) -> List:
		remaining_players = []
		for player in self.players:
			if player.playing_current_round:
				remaining_players.append(player)
		return remaining_players

	def count_remaining_players_in_round(self) -> int: 
		# Helper function to count the total number of players still in the round
		count = 0
		for player in self.players:
			if player.playing_current_round:
				count += 1
		return count
	
	# # def determine_strongest_hand(self): # Move to a separate function evaluator
	# 	# Refer to https://www.poker.org/poker-hands-ranking-chart/. 
	# 	# PRE: This function should only be called after the river has been played
		
	# 	# We update by having the players no longer playing the round
	# 	remaining_players = self.get_remaining_players_in_round()
	# 	for player in remaining_players:
	# 		card_combinations: List[Card] = player.hand + self.community_cards
			
	# 		# Find best possible 5-card combination
	# 		ranks = [card.rank for card in card_combinations]
	# 		suits = [card.suit for card in card_combinations]
			
			
	# 		card_combinations.sort(key=lambda x: x.rank, reverse=True)


	def print_board(self):
		for card in self.community_cards:
			card.print()
		
	def start_new_round(self):
		assert(len(self.players) >= 2) # We cannot start a poker round with less than 2 players...
		# 1. Shuffle, reset pot size 
		self.deck.shuffle()
		self.pot_balance = 0

		# 2. move the dealer position and assign the new small and big blinds
		self.dealer_button_position += 1
		if self.dealer_button_position == len(self.players):
			self.dealer_button_position = 0
		
		# Small Blind
		if self.dealer_button_position + 1 == len(self.players):
			self.players[self.dealer_button_position + 1 - len(self.players)].set_current_bet(self.small_blind)
		else:
			self.players[self.dealer_button_position + 1].set_current_bet(self.small_blind)

		# Big Blind
		if self.dealer_button_position + 2 >= len(self.players):
			self.players[self.dealer_button_position + 2 - len(self.players)].set_current_bet(self.big_blind)
		else:
			self.players[self.dealer_button_position + 2].set_current_bet(self.big_blind)

		# 3. Deal Cards
		# We start dealing with the player directly clockwise of the dealer button
		position_to_deal = self.dealer_button_position + 1 

		for _ in range(2):# Deal the cards to each player, moving in a clockwise circle twice
			for _ in range(len(self.players)): 
				if position_to_deal == len(self.players):
					position_to_deal = 0 # Start at 0 index again

				card = self.deck.draw()
				self.players[position_to_deal].add_card_to_hand(card)
				
				position_to_deal += 1
				
		
		while True:
			# FOR DEBUGGING PURPOSES:
			for idx, player in enumerate(self.players):
				def print_hand(hand: List[Card]): # Helper function
					for card in hand:
						card.print()

				print("Player #", idx, "has the following cards:")
				print_hand(player.hand)

			# 2. Preflop bets, start from the person after the big blind
			# TODO: Consider cases when there are raises
			position_in_play = self.dealer_button_position + 2

			self.min_bet_size = self.big_blind

			for _ in range(len(self.players)):
				if position_in_play >=  len(self.players):
					position_in_play -= len(self.players)

				player = self.players[position_in_play]
				player.update_observed_environment(self)
				player_bet = player.place_bet()
				self.min_bet_size = max(player_bet, self.min_bet_size)
				self.pot_balance += player_bet
				
				if player_bet < self.min_bet_size: # You can't bet less than the bet size
					player.playing_current_round = False
				
				position_in_play += 1 # Go to the next player's turn
				
			if self.count_remaining_players_in_round() == 1:
				self.distribute_pot_to_winning_players()
				break # End round on Pre-Flop

			# 3. Flop
			self.deck.draw() # We must first burn one card, TODO: Show on video
			for _ in range(3): # Draw 3 cards
				self.community_cards.append(self.deck.draw())
			print("The Flop:")
			self.print_board()

			self.play_flop_or_turn_or_river() # Let players play
			if self.count_remaining_players_in_round() == 1:
				self.distribute_pot_to_winning_players()
				break # End round on Flop because everyone else folded
				
			# 4. Turn
			self.deck.draw()# We must first burn one card
			self.community_cards.append(self.deck.draw())
			self.print_board()
			
			self.play_flop_or_turn_or_river()

			if self.count_remaining_players_in_round() == 1:
				self.distribute_pot_to_winning_players()
				break # End round on Turn because everyone else folded
			
			# 5. River
			self.deck.draw()# We must first burn one card
			self.community_cards.append(self.deck.draw())
			self.print_board()

			self.play_flop_or_turn_or_river()

			if self.count_remaining_players_in_round() == 1:
				self.distribute_pot_to_winning_players()
				break # End round on River because everyone else folded
			
			# Less than one player folded, so we need to evaluate the hands and distribute
			
			# TODO: Evalute
			self.distribute_pot_to_winning_players() # There might be more than 1 winner, so in this case, we split the pot evenly
		# End of Round
			

	def play_flop_or_turn_or_river(self): # To make code more efficient
		self.min_bet_size = 0 #  the min-bet size is now 0, since players can simply check
		for player in self.get_remaining_players_in_round():
			player.update_observed_environment(self)
			player_bet = player.place_bet()
			self.min_bet_size = max(player_bet, self.min_bet_size)
			self.pot_balance += player_bet
			
			if player_bet < self.min_bet_size: # You can't bet less than the bet size
				# This is automatically updated at the player level, but just to be safe
				player.playing_current_round = False
		

class Player(): # This is the POV
	def __init__(self, balance) -> None:
		self.hand: List[Card] = [] # The hand is also known as hole cards: https://en.wikipedia.org/wiki/Texas_hold_%27em
		self.player_balance = balance # TODO: Important that this value cannot be modified easily...
		self.current_bet = 0
		
		self.playing_current_round = True
		
		self.observed_env: PokerEnvironment = None

	# Wellformedness, hand is always either 0 or 2 cards
	
	def add_card_to_hand(self, card: Card):
		self.hand.append(card)
		assert(len(self.hand) <= 2)
	
	def clear_hand(self):
		self.hand = []
	
	def set_current_bet(self, bet: int):
		self.current_bet = bet
		self.player_balance -= self.current_bet

	def update_observed_environment(self, env: PokerEnvironment): # Partially observed environment
		self.observed_env = env


	def place_bet(self) -> int:
		action = input("Choose you action ('Check', 'Bet', 'Call', or 'Fold'):")
		if action == "Check":
			if self.observed_env.min_bet_size > 0: 
				print("You cannot check, since there is a bet. You can either 'Bet' or 'Fold'.")
				return self.place_bet()

			else:
				self.set_current_bet(0)

		elif action == "Bet":
			self.set_current_bet(self.observed_env.big_blind)
		
		elif action ==  "Call": # Only Applies to small blind for now, so take away another small blind value
			self.set_current_bet(self.observed_env.small_blind)

		elif action == "Fold":
			self.set_current_bet(0)
			self.playing_current_round = False
		
		else:
			print("Invalid action")
			return self.place_bet()
		
		return self.current_bet



class AIPlayer(Player):
	def __init__(self, balance) -> None:
		super().__init__(balance)
	
	def place_bet(self) -> int: # AI will bet every single round
		self.set_current_bet(self.observed_env.big_blind)
		
		return self.current_bet
	
	
