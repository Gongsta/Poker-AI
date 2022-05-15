# The Poker Environment
import random
from typing import List
from turtle import position

CARD_RANKS = [i for i in range(1, 13)] # Jack = 10, Queen = 11, King = 12, IMPORTANT: Ace = 1
CARD_SUITS = ["Spades", "Clubs", "Hearts", "Diamonds"] 

class Card():
	# Immutable after it has been initialized
	def __init__(self, rank=1, suit="Spades", generate_random=False) -> None:
		self.__rank = rank
		self.__suit = suit
		
		if generate_random: # If we want to just generate a random card
			self.__rank = random.choice(CARD_RANKS)
			self.__suit = random.choice(CARD_SUITS)
	
		# Check validity of card TODO: Maybe put into separate function to check wellformedness
		if self.__suit not in CARD_SUITS: 
			raise Exception("Invalid Suit: {}".format(self.__suit))
		if self.__rank < 1 or self.__rank > 12:
			raise Exception("Invalid Rank: {}".format(self.__rank))

	@property
	def rank(self):
		return self.__rank

	@property
	def suit(self):
		return self.__suit


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
		card_idx = random.randint(0, len(self.__cards) - 1)
		card = self.__cards.pop(card_idx)
		return card
		

class Player():
	def __init__(self, balance) -> None:
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
	
	def set_current_bet(self, bet):
		self.current_bet = bet

	def place_bet(self) -> int:


	

	# Options: Check, Bet, Raise, Fold, 
		

class AIPlayer(Player):
	def __init__(self) -> None:
		super().__init__()
	
	

class PokerEnvironment():
	def __init__(self) -> None:
		self.players: List[Player] = []
		self.deck = Deck() 
		
		# Changes every round
		self.dealer_button_position = 0 # This button will move every round
		self.pot_balance = 0 # keep track of the current pot size
		self.community_cards: List[Card] = []
		self.min_bet_size = 0

		
		# These values should rarely change
		self.new_player_balance = 10000
		self.small_blind = 50 
		self.big_blind = 100 # TODO: Check if this is even necessary 
	
	def add_player(self):
		self.players.append(Player(balance=self.new_player_balance))
	
	def get_winning_players(self) -> List[Player]:
		# If there is more than one winning player, the pot is split. We assume that we only run things once
		winning_players: List[Player] = []
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


	def get_remaining_players_in_round(self) -> List[Player]:
		remaining_players = []
		for player in self.players:
			if player.playing_current_round:
				remaining_players.append(self.players)
		return remaining_players

	def count_remaining_players_in_round(self) -> int: 
		# Helper function to count the total number of players still in the round
		count = 0
		for player in self.players:
			if player.playing_current_round:
				count += 1
		return count


	def start_new_round(self):
		assert(len(self.players) >= 2) # We cannot start a poker round with less than 2 players...
		# 1. Shuffle, reset pot size 
		self.deck.shuffle()
		self.pot_balance = 0

		# 2. move the dealer position and assign the new small and big blinds
		self.dealer_button_position += 1
		if self.dealer_button_position == len(self.players):
			self.dealer_button_position = 0
		
		if self.dealer_button_position + 1 == len(self.players):
			self.players[self.dealer_button_position + 1 - len(self.players)].set_current_bet(self.small_blind)
			
		if self.dealer_button_position + 2 >= len(self.players):
			self.players[self.dealer_button_position + 2 - len(self.players)].set_current_bet(self.big_blind)

		# 3. Deal Cards
		# We start dealing with the player directly clockwise of the dealer button
		position_to_deal = self.dealer_button_position + 1 

		for _ in range(2):# Deal the cards to each player
			for _ in range(len(self.players)): 
				if position_to_deal == len(self.players):
					position_to_deal = 0 # Start at 0 index again

				card = self.deck.draw()
				self.players[position_to_deal].add_card_to_hand(card)
				
		
		# FOR DEBUGGING PURPOSES:
		for player in self.players:
			print(player.hand)

		# 2. Preflop bets, start from the person after the big blind
		# TODO: Consider cases when there are raises
		self.min_bet_size = self.big_blind
		for player in self.players:
			player_bet = player.place_bet()
			self.min_bet_size = max(player_bet, self.min_bet_size)
			self.pot_balance += player_bet
			
			if player_bet < self.min_bet_size: # You can't bet less than the bet size
				player.playing_current_round = False
			
			
		if self.count_remaining_players_in_round() == 1:
			self.distribute_pot_to_winning_players()
		# 3. Flop
		self.min_bet_size = 0
		self.deck.draw() # We must first burn one card
		for _ in range(3): # Draw 3 cards
			self.community_cards.append(self.deck.draw())
			
		
		for player in self.get_remaining_players_in_round():
			self.min_bet_size = max(player_bet, self.min_bet_size)
			self.pot_balance += player_bet
			
			if player_bet < self.min_bet_size: # You can't bet less than the bet size
				player.playing_current_round = False
		
		if self.count_remaining_players_in_round() == 1:
			self.distribute_pot_to_winning_players()
			
		# 4. Turn
		self.play_turn_or_river()
		
		# 5. River
		self.play_turn_or_river()
		
		
		self.distribute_pot_to_winning_players() # There might be more than 1 winner, so in this case, we split the pot evenly
		# End of Round
			

		def play_turn_or_river(self): # To make code more efficient
			self.deck.draw()# We must first burn one card
			self.community_cards.append(self.deck.draw())
			for player in self.get_remaining_players_in_round():
				self.min_bet_size = max(player_bet, self.min_bet_size)
				self.pot_balance += player_bet
				
				if player_bet < self.min_bet_size: # You can't bet less than the bet size
					player.playing_current_round = False
		
		if self.count_remaining_players_in_round() == 1:
			self.distribute_pot_to_winning_players()
