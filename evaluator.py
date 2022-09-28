# Uses Cactus Kev’s 5-Card Evaluator: http://suffe.cool/poker/evaluator.html
# I am aware that this method is not the most efficient, I will look into implementations if needed

# https://www.codingthewheel.com/archives/poker-hand-evaluator-roundup

"""
# Some thoughts, the evaluator is sgoing to be important because it determines the rules
# of the game. However, we won't give our AI this. The environment will simply 
# feed the final reward, which is win or not win. 

# Reward = the amount of money you win or lose. 
# The optimal policy probably has to be probabilistic, actually no... maybe 
because the opponent will start figuring it out and calling / folding every time. 

Also, what kind of opponents do we want to be playing? 

Card representation also needs to be fast. Everything needs to be fast, for the AI
to run lots of simulations and figure out the optimal strategy. 

# I think I will consider two approaches
- Card Kev's which is a 5-card lookup table, will need to do lookup 21 times...
- Two + Two algorithm which is for a 7-card lookup table
https://github.com/chenosaurus/poker-evaluator/

Speed is pretty important, since I want to train the AI as fast as possible, so that it learns the optimal 
policy. If the game is slow, then there is no point. 
"""

# Representation is key to performance. This is going to be terrifying, as I am going to be working with bits..
from typing import List
import random
from table import generate_table

BIT_POSITION_TABLE = generate_table()

CARD_SUITS = ["Clubs", "Diamonds", "Hearts","Spades"] 
CARD_SUITS_DICT = {"Clubs": 0, "Diamonds": 1, "Hearts": 2,"Spades": 3}

BIT_MASK_1 = int('0x11111111111111', 16) # 0x111...1
BIT_MASK_2 = int('0x22222222222222', 16) # 0x222...2
BIT_MASK_4 = int('0x44444444444444', 16) # 0x444...4
BIT_MASK_8 = int('0x88888888888888', 16) # 0x888...8

BIT_MASKS = [BIT_MASK_1, BIT_MASK_2, BIT_MASK_4, BIT_MASK_8]
""" For CARDS_BIT_SUITS_DICT, we have
0001 (1) -> Clubs
0010 (2) -> Diamonds
0100 (4) -> Hearts
1000 (8) -> Spades
"""
CARD_BIT_SUITS_DICT = {1: "Clubs", 2: "Diamonds", 4: "Hearts", 8: "Spades"}

CARD_RANKS = [i for i in range(2, 15)] # Jack = 11, Queen = 12, King = 13, IMPORTANT: Ace = 14 since we use that for sorting
CARD_SUITS = ["Clubs", "Diamonds", "Hearts","Spades"] 

RANK_KEY = {"A": 14, "2": 2, "3": 3, "4":4, "5":5, "6":6, # Supports both "T" and "10" as 10
			"7": 7, "8": 8, "9": 9, "T": 10, "10":10, "J": 11, "Q": 12, "K":13} 

# INVERSE_RANK_KEY = {14: "A", 2: "02", 3: "03", 4:"04", 5:"05", 6:"06",
# 			7:"07", 8:"08", 9:"09", 10:"10", 11: "J", 12: "Q", 13: "K"}

INVERSE_RANK_KEY = {14: "A", 2: "2", 3: "3", 4:"4", 5:"5", 6:"6",
			7:"7", 8:"8", 9:"9", 10:"T", 11: "J", 12: "Q", 13: "K"}

SUIT_KEY = {"c": "Clubs", "d": "Diamonds", "h":"Hearts","s": "Spades"}


class Card():
	# Immutable after it has been initialized
	def __init__(self,rank=14, suit="Spades", rank_suit=None, generate_random=False) -> None:

		if rank_suit: # Ex: "KD" (King of diamonds), "10H" (10 of Hearts),
			self.__rank = RANK_KEY[rank_suit[:-1]]
			self.__suit = SUIT_KEY[rank_suit[-1].lower()]

		else:
			self.__rank = rank
			assert(self.__rank >= 2 and self.__rank <= 14)
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
	
	@property
	def idx(self):
		"""
		Used for the RL part. We will represent the hand as a list of 52 binary integers.
		[AC, AD, AH, AS, 2C, 2D, ... KH, KS]
		0 .  1 . 2 . 3 . 4 . 5 .     50, 51  
		"""
		rank = self.__rank
		if self.__rank == 14: # for the aces
			rank = 1
		rank -= 1
		return rank*4 + CARD_SUITS_DICT[self.__suit]
	
	def __str__(self): # Following the Treys format of printing
		return INVERSE_RANK_KEY[self.rank] + self.suit[0].lower()
	

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
		
ACTIONS = ["Fold", "Call", "Raise"]



class CombinedHand:
	def __init__(self, hand: List [Card]=[]):
		self.hand: List[Card] = hand
		self.hand_strength = 0
		self.h = 0 
		self.comparator = 0
		
		if hand != None:
			self.update_binary_representation()
		
	def __str__(self):
		s = ""
		for h in self.hand:
			s += str(h) + ", "
		
		return s
	
	def as_list(self):
		# Save hand as a linary of characters
		s = []
		for h in self.hand:
			s.append(str(h))
		
		return s
		

	def __len__(self):
		return len(self.hand)

	def update_binary_representation(self):
		self.h = 0
		for card in self.hand: # Convert cards into our binary representation
			self.h += 1 << int(4 * (card.rank - 1)) << CARD_SUITS_DICT[card.suit] # TODO: I can probs optimize by storing the multiplication in another CARDS_RANK_DICT table

			if card.rank == 14: # For aces, we need to add them at the beginning as well
				self.h += 1 << CARD_SUITS_DICT[card.suit]

	def add_combined_hands(self, *hands):
		for hand in hands:
			for card in hand.hand:
				self.hand.append(card)

		self.update_binary_representation()

	def add_cards(self, *cards):
		for card in cards:
			self.hand.append(card)
			
		self.update_binary_representation()

	def get_binary_representation(self):
		return bin(self.h)
		
	def get_hand_strength(self, verbose=False):
		# In case of ties, we set self.comparator:
		# 1 (Royal Flush) - Always Tie
		# 2 (Straight Flush) - Set self.comparator = [lowest_straight_flush]
		# 3 (Four of A kind) - Set self.comparator = [four_of_a_kind, kicker]
		# 4 (Full House) - self.comparator =  [three_of_a_kind, two_of_a_kind]
		# 5 (Flush) - self.comparator = [flush1, flush2, flush3, flush4, flush5]
		# 6 (Straight) - self.comparator = lowest_straight
		# 7 (Three of a kind) - self.comparator = [three_of_a_kind, kicker1, kicker2]
		# 8 (Two-Pair) - self.comparator= [Rank1, Rank2, kicker]
		# 9 (Pair) - self.comparator = [pair, kicker1, kicker2, kicker3]
		# 10 (High Card) - self.comparator = [kicker1, kicker2, kicker3, kicker4, kicker5]

		# 1 - Royal Flush
		h = self.h
		royal_flush = (h >> 36) & (h >> 40) & (h >> 44) & (h >> 48) & (h >> 52)
		if royal_flush:
			if verbose:
				print("Royal Flush of", CARD_BIT_SUITS_DICT[royal_flush])
			self.hand_strength = 1
			return
		
		# 2 - Straight Flush
		h = self.h
		hh = (h) & (h >> 4) & (h >> 8) & (h >> 12) & (h >> 16)
		
		if hh:
			highest_low_card = 0
			checker = hh
			for i in range(1,11):
				if (checker & 15):
					highest_low_card = i
				checker = checker >> 4

			self.hand_strength = 2
			self.comparator = [highest_low_card]
			if verbose:
				print("Straight Flush starting with :",  self.comparator[0]) 
			return
			# If TIE, you can just use hh to compare
		
		# 3 - Four of A Kind
		h = self.h >> 4 # Ignore the first 4 aces
		hh = (h) & (h >> 1) & (h >> 2) & (h >> 3) & BIT_MASK_1
		if hh:
			four_of_a_kind = BIT_POSITION_TABLE[hh]//4 + 2
			self.hand_strength = 3
			kicker = 0
			for card in self.hand:
				if (card.rank != four_of_a_kind):
					kicker = max(kicker, card.rank)
			
			self.comparator = [four_of_a_kind,kicker]
			if verbose:
				print("Four of a kind: ", self.comparator[0], "Kicker: ", self.comparator[1])  # hh is guaranteed to only have a single "1" bit
			return

		
		# 4 - Full House
		threes, threes_hh = self.check_threes() 
		twos = self.check_twos(threes_hh) # Exclusive pairs, not threes, needed for full house
		if (len(threes) >= 1 and len(twos) >= 1) or len(threes) > 1:
			self.hand_strength = 4

			if (len(threes) > 1): # Edge case when there are two trips
				# Search for largest pair
				max_three = max(threes)
				if (len(twos) == 0):
					max_two = 0
				else:
					max_two = max(twos)

				for three in threes:
					if (three != max_three):
						max_two = max(max_two, three)
				self.comparator = [max_three, max_two]

			else: # Regular Case
				self.comparator = [max(threes), max(twos)]
			
			if verbose:
				print("Full house with threes of: {}, pair of: {}".format(self.comparator[0], self.comparator[1]))
			return

		# 5 - Flush
		h = self.h >> 4 # Ignore the right most aces
		for idx, MASK in enumerate(BIT_MASKS):
			hh = h & MASK
			if bin(hh).count("1") >= 5:
				suit = CARD_SUITS[idx]
				final_hand = []
				for card in self.hand:
					if (card.suit == suit):
						final_hand.append(card.rank)
				
				final_hand = sorted(final_hand, reverse=True)[:5] # Sort from best to worst
				self.hand_strength = 5

				self.comparator = final_hand
				if verbose:
					print("Flush with hand: ",self.comparator) 
				return

		# 6 - Straight
		h = self.h
		hh1 = h & BIT_MASK_1
		hh1 = (hh1) | (hh1 << 1) | (hh1 << 2) | (hh1 << 3)
		hh2 = h & BIT_MASK_2
		hh2 = (hh2) | (hh2 >> 1) | (hh2 << 1) | (hh2 << 2)	
		hh4 = h & BIT_MASK_4
		hh4 = (hh4) | (hh4 << 1) | (hh4 >> 1) | (hh4 >> 2)	
		hh8 = h & BIT_MASK_8
		hh8 = (hh8) | (hh8 >> 1) | (hh8 >> 2) | (hh8 >> 3)
		hh = hh1 | hh2 | hh4 | hh8
		hh = (hh) & (hh >> 4) & (hh >> 8) & (hh >> 12) & (hh >> 16)
		
		if hh:
			low_card = 1
			curr = 1
			n = hh 
			while (curr < 15):
				if (n & 1):
					low_card = curr
				
				curr += 1
				n = n >> 4

			self.hand_strength = 6
			self.comparator = [low_card]
			if verbose:
				print("Straight starting from: ", self.comparator[0])
			return low_card
			
			
		# 7 - Three of A Kind
		# threes = self.check_threes() # This is already ran in the full house
		if len(threes) == 1: # If more then 1 trips, we would have covered the case in the full-house
			self.hand_strength = 7
			kickers = []
			for card in self.hand:
				if (card.rank != threes[0]):
					kickers.append(card.rank)
			kickers.sort(reverse=True)
			self.comparator = [threes[0], kickers[0], kickers[1]]
			if verbose:
				print("Three of a kind: ", self.comparator[0], "Kickers: ", self.comparator[1:]) #TODO: Check Value
			return
		
		# 8 - Two Pair / 9 - One Pair
		# twos = self.check_threes() # This is already ran in the full house
		if len(twos) >= 1:	# Move this for comparison?
			twos.sort(reverse=True)
			if len(twos) >= 2: # Two Pair
				self.hand_strength = 8
				kicker = 0
				for card in self.hand:
					if (card.rank != twos[0] and card.rank != twos[1]):
						kicker = max(kicker, card.rank)
				self.comparator = [twos[0], twos[1], kicker]
				if verbose:
					print("Two Pair: ", self.comparator[0], ", ", self.comparator[1],  "Kicker: ", self.comparator[2]) #TODO: Check Value
			else: # One Pair
				self.hand_strength = 9
				kickers = []
				for card in self.hand:
					if (card.rank != twos[0]):
						kickers.append(card.rank)
				kickers.sort(reverse=True)
				self.comparator = [twos[0], kickers[0], kickers[1], kickers[2]]
				if verbose:
					print("One Pair: ", self.comparator[0], "Kickers: ", self.comparator[1:]) #TODO: Check Value

			return
			

		# 10 - High Card
		self.hand_strength = 10
		kickers = []
		for card in self.hand:
				kickers.append(card.rank)
		self.comparator = sorted(kickers, reverse=True)[:5] # From best to worst ranks
		if verbose:
			print("High Card: ", self.comparator[-1], "Kickers: ", self.comparator[:4])
		return
		

	def check_threes(self):
		h = self.h >> 4 # Ignore right most aces
		hh = (((h) & (h >> 1) & (h >> 2)) | ((h >> 1) & (h >> 2) & (h >> 3)) | ((h) & (h >> 1) & (h >> 3)) | ((h) & (h >> 2) & (h >> 3))) & BIT_MASK_1
		
		threes = []
		if hh:
			low_card = 2
			n = hh 
			while True:
				if (n & 1):
					threes.append(low_card)
				
				if (low_card >= 14): #Exit loop when we reached last card
					break
				low_card += 1
				n = n >> 4
			
		# No Guarantee that hh only has 1 bit, but the bit will always be on every 4th
		return threes, hh

	def check_twos(self, threes_hh):
		h = self.h >> 4 # Ignore right most aces
		hh = (((h) & (h >> 1)) | ((h) & (h >> 2)) | ((h) & (h >> 3)) | ((h >> 1) & (h >> 2)) | ((h >> 1) & (h >> 3)) | ((h >> 2) & (h >> 3))) & BIT_MASK_1
		hh = hh ^ threes_hh
		twos = []
		if hh:
			low_card = 2
			n = hh 
			while True:
				if (n & 1):
					twos.append(low_card)
				
				if (low_card >= 14): #Exit loop when we reached last card
					break
				low_card += 1
				n = n >> 4
			
		return twos
	

class Evaluator:
	def __init__(self):
		self.hands: List[CombinedHand] = []
	
	def add_hands(self, *combined_hands: CombinedHand):
		for combined_hand in combined_hands:
			self.hands.append(combined_hand)
	
	def clear_hands(self):
		self.hands = []
	
	def __str__(self):
		ans = ""
		for hand in self.hands:
			ans += str(hand) + " "
			ans += '\n'
		return ans
	
	def get_winner(self) -> List[int]: # Return a list of 0-indexed of players who won the pot. If multiple, then split
		for hand in self.hands:
			hand.get_hand_strength()
		hand_strengths = [hand.hand_strength for hand in self.hands]
		best_hand_val = min(hand_strengths)
		potential_winners = [i for i, x in enumerate(hand_strengths) if x == best_hand_val]
		
		# TODO: Idea to optimize in the future, just make the best hand as a list, and then compare if necessary.
		
		if len(potential_winners) > 1: # Potential ties
			if best_hand_val == 1: # Royal Flush, Automatic Tie
				return potential_winners

			elif best_hand_val == 2: # Straight Flush, check low card
				highest_low_card = 0
				for winner in potential_winners:
					highest_low_card = max(highest_low_card, self.hands[winner].comparator[0])
				winners = []
				for winner in potential_winners:
					if (self.hands[winner].comparator[0] == highest_low_card):
						winners.append(winner)
				return winners

			elif best_hand_val == 3: # Four of a kind
				highest_four = 0
				highest_kicker = 0
				for winner in potential_winners:
					highest_four = max(highest_four, self.hands[winner].comparator[0])
					highest_kicker = max(highest_kicker, self.hands[winner].comparator[1])
				
				winners = []
				for winner in potential_winners:
					if (self.hands[winner].comparator[0] == highest_four and self.hands[winner].comparator[1] == highest_kicker):
						winners.append(winner)
				
				return winners

			elif best_hand_val == 4: # Full House
				highest_threes = 0
				highest_twos = 0
				for winner in potential_winners:
					highest_threes = max(highest_threes, self.hands[winner].comparator[0])
					highest_twos = max(highest_twos, self.hands[winner].comparator[1])
				
				winners = []
				for winner in potential_winners:
					if (self.hands[winner].comparator[0] == highest_threes and self.hands[winner].comparator[1] == highest_twos): # Pick player with best full house
						winners.append(winner)
				
				if (len(winners) ==0): # Edge case when we have full house over full house
					for winner in potential_winners:
						if (self.hands[winner].comparator[0] == highest_threes):
							winners.append(winner)
				return winners

			elif best_hand_val == 5: # Flush
				best_flush = [0,0,0,0,0]
				
				# Check from best card to worst card. 
				for i in range(5):
					for winner in potential_winners:
						best_flush[i] = max(best_flush[i], self.hands[winner].comparator[i])

					winners = []
					for winner in potential_winners:
						if (self.hands[winner].comparator[i] == best_flush[i]):
							winners.append(winner)

					if (len(winners) == 1):  #  Whenever there is only 1 winner, just return
						return winners
				
				return winners

			elif best_hand_val == 6: # Straight
				highest_low_card = 0
				for winner in potential_winners:
					highest_low_card = max(highest_low_card, self.hands[winner].comparator[0])
				
				winners = []
				for winner in potential_winners:
					if (highest_low_card == self.hands[winner].comparator[0]):
						winners.append(winner)
						
				return winners
				
			elif best_hand_val == 7: # Three of a kind
				best_hand = [0,0,0] # [three_of_a_kind, kicker1, kicker2]
				for i in range(3):
					for winner in potential_winners:
						best_hand[i] = max(best_hand[i], self.hands[winner].comparator[i])
						
					winners = []
					for winner in potential_winners:
						if (self.hands[winner].comparator[i] == best_hand[i]): winners.append(winner)
					if len(winners) == 1:
						return winners
				
				return winners # In case of tie, this will be run

			elif best_hand_val == 8: # Two Pair
				best_hand = [0,0,0] # [best_pair1, best_pair2, kicker]
				for i in range(3):
					for winner in potential_winners:
						best_hand[i] = max(best_hand[i], self.hands[winner].comparator[i])
						
					winners = []
					for winner in potential_winners:
						if (self.hands[winner].comparator[i] == best_hand[i]): winners.append(winner)
					if len(winners) == 1:
						return winners
				
				return winners # In case of tie, this will be run
					
			elif best_hand_val == 9: # One Pair
				best_hand = [0,0,0,0] # [pair, kicker1, kicker2, kicker3]
				for i in range(4):
					for winner in potential_winners:
						best_hand[i] = max(best_hand[i], self.hands[winner].comparator[i])
						
					winners = []
					for winner in potential_winners:
						if (self.hands[winner].comparator[i] == best_hand[i]): winners.append(winner)
					if len(winners) == 1:
						return winners
				
				return winners # In case of time, this will be run
			elif best_hand_val == 10: # High Card
				best_hand = [0,0,0,0,0] # [kicker1, kicker2, kicker3, kicker4, kicker5]
				for i in range(5):
					for winner in potential_winners:
						best_hand[i] = max(best_hand[i], self.hands[winner].comparator[i])
						
					winners = []
					for winner in potential_winners:
						if (self.hands[winner].comparator[i] == best_hand[i]): winners.append(winner)
					if len(winners) == 1:
						return winners

				return winners

			
		else: # Single person has the best hand
			return potential_winners
		