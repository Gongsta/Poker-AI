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
from environment import *
import numpy as np
from table import generate_table

BIT_POSITION_TABLE = generate_table()

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


class CombinedHand:
	def __init__(self, hand: List [Card]):
		self.hand = hand
		self.hand_strength = 0
		self.h = 0 
		self.hh = 0
		self.kicker = 0
		for card in hand: # Convert cards into our binary representation
			self.h += 1 << int(4 * (card.rank - 1)) << CARD_SUITS_DICT[card.suit] # TODO: I can probs optimize by storing the multiplication in another CARDS_RANK_DICT table

			if card.rank == 14: # For aces, we need to add them at the beginning as well
				self.h += 1 << CARD_SUITS_DICT[card.suit]
		


	def get_binary_representation(self):
		return bin(self.h)
	
	def get_hand_strength(self, verbose=False):
		# There is some inconsistency in the return format, so just to be aware here, how it works
		# The reason I am not returning simply the best hand is because further comparison is needed down the line, for kickers.
		# Maybe I can just code in the kickers here? Return [Card1, Card2, Card3, Card4, Card5]?
		# 1 (Royal Flush) - dont care, returns 4-bit
		# 2 (Straight Flush) - returns 40-bit integer
		# 3 (Four of A kind) - returns a 49-bit integer, (maybe List + kicker here? Doesn't make it much slower)
		# 4
		# 5
		# 6
		# 7
		# 8 (Two-Pair) - Returns [Rank1, Rank2]
		# 9 (Pair) - Returns [Rank]
		# 10 (Pair) - Returns 56-bit		

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
			if verbose:
				print("Straight Flush starting with the lowest card of",  ) # TODO
			self.hand_strength = 2
			return hh 
			# If TIE, you can just use hh to compare
		
		# 3 - Four of A Kind
		h = self.h >> 4 # Ignore the first 4 aces
		hh = (h) & (h >> 1) & (h >> 2) & (h >> 3) & BIT_MASK_1
		if hh:
			if verbose:
				print("Four of a kind: ", BIT_POSITION_TABLE[hh]//4 + 2)  # hh is guaranteed to only have a single "1" bit
			self.hand_strength = 3
			return hh
			# TODO: If TIE, you need to use `self.h` and find the kicker...

		
		# 4 - Full House
		threes, threes_hh = self.check_threes() 
		twos = self.check_twos(threes_hh) # Exclusive pairs, not threes, needed for full house
		if len(threes) >= 1 and len(twos) >= 1:
			self.hand_strength = 4
			return [max(threes), max(twos)] 

		# 5 - Flush
		h = self.h >> 4 # Ignore the right most aces
		for MASK in BIT_MASKS:
			hh = h & MASK
			if bin(hh).count("1") >= 5:
				# print("Flush with hand: ", bin(hh)) #TODO: Get the card values
				self.hand_strength = 5
				return hh
				# TODO: If tie, you can just use hh, finding the value of the cards

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
			n = hh 
			while True:
				if (n & 1):
					if verbose:
						print("Straight with lowest card: ", low_card)
					break
				
				low_card += 1
				n = n >> 4

			self.hand_strength = 6
			return low_card
			
			
		# 7 - Three of A Kind
		# threes = self.check_threes() # This is already ran in the full house
		if len(threes) >= 1:	# Move this for comparison?
			if verbose:
				print("Three of a kind: ", threes) #TODO: Check Value

			self.hand_strength = 7
			return threes
		
		# 8 - Two Pair / 9 - One Pair
		# twos = self.check_threes() # This is already ran in the full house
		if len(twos) >= 1:	# Move this for comparison?
			if verbose:
				if len(twos) >= 2: # Two Pair
					print("One Pair: ", twos) #TODO: Check Value
				else: # One Pair
					print("One Pair:", twos) #TODO: Check Value

			if len(twos) >= 2: # Two Pair
				self.hand_strength = 8
			else: # One Pair
				self.hand_strength = 9
			
			return twos
			

		# 10 - High Card
		self.hand_strength = 10
		return self.h # Just return the original self.h
		

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
		self.hands: List[CombinedHand] = None
	
	def add_hand(self, combined_hand: CombinedHand):
		self.hands.append(combined_hand)
	
	def get_winner(self): # Return a list of index of players who won the pot. If multiple, then split
		for hand in self.hands:
			hand.get_hand_strength()
		hand_strengths = [hand.hand_strength for hand in self.hands]
		best_hand_val = min(hand_strengths)
		potential_winners = [i for i, x in enumerate(hand_strengths) if x == best_hand_val]
		
		if len(potential_winners) > 1: # Potential ties
			if best_hand_val == 1: # Royal Flush, Automatic Tie
				return potential_winners

			elif best_hand_val == 2: # Straight Flush, check low card

			elif best_hand_val == 3:
			elif best_hand_val == 2:
			elif best_hand_val == 2:
			elif best_hand_val == 2:
			elif best_hand_val == 2:
			elif best_hand_val == 2:

			for i in potential_winners:
				return 0
			
		else: # Single person has the best hand
			return potential_winners
		