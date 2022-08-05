import unittest
import os
import sys
import shutil


if __name__ == "__main__":
	devpath = os.path.relpath(os.path.join('..'), start=os.path.dirname(__file__))
	sys.path = [devpath] + sys.path

# Import Libraries
from environment import *
from evaluator import *

class UnitTests(unittest.TestCase):
	
	# Unit Testing
	def test_card_initialization(self):
		jackOfHearts = Card(11, "Hearts")
		self.assertEqual(jackOfHearts.rank, 11)
		self.assertEqual(jackOfHearts.suit, "Hearts")
		
		with self.assertRaises(Exception): # Invalid Rank
			Card(15, "Diamonds")
		with self.assertRaises(Exception): # Invalid Rank
			Card(1, "Diamonds")
		with self.assertRaises(Exception): # Invalid Suit
			Card(2, "B")
		
		AceOfClubs = Card(rank_suit="AC")
		self.assertEqual(AceOfClubs.rank, 14)
		self.assertEqual(AceOfClubs.suit, "Clubs")

	def test_deck_initalization(self):
		new_deck = Deck()
		assert(new_deck.total_remaining_cards == 52)
		
	def test_combinedHand(self):
		a = Card(rank_suit="2C")
		b = Card(rank_suit="2D")
		c = Card(rank_suit="2H")
		d = Card(rank_suit="2S")
		hand = [a, b, c, d]
		evaluator = CombinedHand(hand)
		self.assertEqual(evaluator.get_binary_representation(), '0b11110000')

		a = Card(rank_suit="AC")
		b = Card(rank_suit="AD")
		c = Card(rank_suit="AH")
		d = Card(rank_suit="AS")
		hand = [a, b, c, d]
		evaluator = CombinedHand(hand)
		self.assertEqual(evaluator.get_binary_representation(), '0b11110000000000000000000000000000000000000000000000001111')

		a = Card(rank_suit="2C")
		b = Card(rank_suit="3C")
		c = Card(rank_suit="4C")
		d = Card(rank_suit="5C")
		hand = [a, b, c, d]
		evaluator = CombinedHand(hand)
		self.assertEqual(evaluator.get_binary_representation(), '0b10001000100010000')

		a = Card(rank_suit="AC")
		b = Card(rank_suit="2S")
		c = Card(rank_suit="4S")
		d = Card(rank_suit="5S")
		hand = [a, b, c, d]
		evaluator = CombinedHand(hand)
		
		final_string = '0b'
		final_string += '1' 
		final_string += '0000' * 8
		final_string += '10001000000010000001'
		self.assertEqual(evaluator.get_binary_representation(), final_string)
		
	def test_royal_flush(self):
		# Royal Flush of Hearts
		a = Card(rank_suit="AH")
		b = Card(rank_suit="KH")
		c = Card(rank_suit="JH")
		d = Card(rank_suit="QH")
		e = Card(rank_suit="10H")
		f = Card(rank_suit="9S") 
		g = Card(rank_suit="2C")
		h = Card(rank_suit="3C")
		hand1 = [a,b,c,d,e,f,g]
		hand2 = [a,b,c,d,e,f,h]
		hand1 = CombinedHand(hand1)
		hand2 = CombinedHand(hand2)
		hand1.get_hand_strength()
		hand2.get_hand_strength()
		
		self.assertEqual(hand1.hand_strength, 1)
		self.assertEqual(hand2.hand_strength, 1)

		evaluator = Evaluator()
		evaluator.add_hands(hand1)
		evaluator.add_hands(hand2)

		self.assertEqual(evaluator.get_winner(), [0,1])
	
	def test_royal_flush2(self):
		# Royal Flush of Hearts
		a = Card(rank_suit="AH")
		b = Card(rank_suit="KH")
		c = Card(rank_suit="JH")
		d = Card(rank_suit="QH")
		e = Card(rank_suit="10H")
		f = Card(rank_suit="9S") 
		g = Card(rank_suit="2C")

		hand1 = [a,b,c,d,e,f,g]
		# Royal Flush of Spades (this wouldn't happen in a real match)
		a = Card(rank_suit="AS")
		b = Card(rank_suit="KS")
		c = Card(rank_suit="JS")
		d = Card(rank_suit="QS")
		e = Card(rank_suit="10S")
		f = Card(rank_suit="9S") 
		g = Card(rank_suit="2C")
		hand2 = [a,b,c,d,e,f,g]
		hand1 = CombinedHand(hand1)
		hand2 = CombinedHand(hand2)
		evaluator = Evaluator()
		evaluator.add_hands(hand1)
		evaluator.add_hands(hand2)
		self.assertEqual(evaluator.get_winner(), [0,1])

	def test_straight_flush(self):
		a = Card(rank_suit="4S")
		b = Card(rank_suit="5S")
		c = Card(rank_suit="JH")
		d = Card(rank_suit="6S")
		e = Card(rank_suit="10H")
		f = Card(rank_suit="7S") 
		g = Card(rank_suit="8S")
		h = Card(rank_suit="9S")
		hand1 = [a,b,c,d,e,f,g]
		hand2 = [h,b,c,d,e,f,g]
		hand3 = [a,b,c,d,e,f,g]
		hand1 = CombinedHand(hand1)
		hand2 = CombinedHand(hand2)
		hand1.get_hand_strength()
		hand2.get_hand_strength()
		self.assertEqual(hand1.hand_strength, 2)
		self.assertEqual(hand2.hand_strength, 2)

		# No tie, since hand2 has a better straight flush
		evaluator = Evaluator()
		evaluator.add_hands(hand1)
		evaluator.add_hands(hand2)
		self.assertEqual(evaluator.get_winner(), [1])
		
		# Tie, since both hands have the same straight flush
		evaluator.clear_hands()
		evaluator.add_hands(hand1)
		evaluator.add_hands(hand3)
		self.assertEqual(evaluator.get_winner(), [0, 1])

	def test_four_of_a_kind(self):
		a = Card(rank_suit="4S")
		b = Card(rank_suit="6C")
		c = Card(rank_suit="JH")
		d = Card(rank_suit="6S")
		e = Card(rank_suit="10H")
		f = Card(rank_suit="6D") 
		g = Card(rank_suit="6H")
		hand1 = [a,b,c,d,e,f,g]
		hand1 = CombinedHand(hand1)
		hand1.get_hand_strength()
		self.assertEqual(hand1.hand_strength, 3)
	
		# Check for ties
		a = Card(rank_suit="4S")
		b = Card(rank_suit="AC")
		c = Card(rank_suit="JH")
		d = Card(rank_suit="AS")
		e = Card(rank_suit="10H")
		f = Card(rank_suit="AD") 
		g = Card(rank_suit="AH")
		hand2 = [a,b,c,d,e,f,g]
		hand2 = CombinedHand(hand2)
		hand2.get_hand_strength()
		self.assertEqual(hand2.hand_strength, 3)
		
		evaluator = Evaluator()
		evaluator.add_hands(hand1)
		evaluator.add_hands(hand2)
		self.assertEqual(evaluator.get_winner(), [1])

		evaluator.clear_hands()
		# Case for tie
		evaluator.add_hands(hand2, hand2)
		self.assertEqual(evaluator.get_winner(), [0,1])
		
	
	def test_full_house(self):
		a = Card(rank_suit="4S")
		b = Card(rank_suit="4C")
		c = Card(rank_suit="4H")
		d = Card(rank_suit="2S")
		e = Card(rank_suit="2H")
		f = Card(rank_suit="AD") 
		g = Card(rank_suit="7H")
		hand1 = [a,b,c,d,e,f,g]
		hand1 = CombinedHand(hand1)
		hand1.get_hand_strength()
		self.assertEqual(hand1.hand_strength, 4)
		
		
		# Better Pair of threes (10s vs 4s)
		a = Card(rank_suit="10S")
		b = Card(rank_suit="10C")
		c = Card(rank_suit="10H")
		hand2 = [a,b,c,d,e,f,g]
		hand2 = CombinedHand(hand2)
		evaluator = Evaluator()
		evaluator.add_hands(hand1)
		evaluator.add_hands(hand2)
		self.assertEqual(evaluator.get_winner(), [1])


		# Same Pair of Threes, better pair of twos
		d = Card(rank_suit="3S")
		e = Card(rank_suit="3H")
		hand3 = [a,b,c,d,e,f,g]
		hand3 = CombinedHand(hand3)

		evaluator.clear_hands()
		evaluator.add_hands(hand2, hand3)
		self.assertEqual(evaluator.get_winner(), [1])
		
		evaluator.clear_hands()
		evaluator.add_hands(hand3, hand3)
		self.assertEqual(evaluator.get_winner(), [0, 1])
		
		
		# Edge Case: Two pairs of threes should still give full house
		a = Card(rank_suit="4S")
		b = Card(rank_suit="4C")
		c = Card(rank_suit="4H")
		d = Card(rank_suit="2S")
		e = Card(rank_suit="2H")
		f = Card(rank_suit="2D") 
		g = Card(rank_suit="7H")
		hand1 = [a,b,c,d,e,f,g]
		hand1 = CombinedHand(hand1)
		hand1.get_hand_strength()
		self.assertEqual(hand1.hand_strength, 4)
		
	def test_flush(self):
		a = Card(rank_suit="4S")
		b = Card(rank_suit="AS")
		c = Card(rank_suit="JS")
		d = Card(rank_suit="AD")
		e = Card(rank_suit="10H")
		f = Card(rank_suit="8S") 
		g = Card(rank_suit="7S")
		hand1 = [a,b,c,d,e,f,g]
		hand1 = CombinedHand(hand1)
		hand1.get_hand_strength()
		self.assertEqual(hand1.hand_strength, 5)
		
		b = Card(rank_suit="3S") # Worse flush
		hand2 = [a,b,c,d,e,f,g]
		evaluator = Evaluator()
		evaluator.add_hands(hand1, hand2)
		self.assertEqual(evaluator.get_winner(), [0])


	def test_straight(self):
		a = Card(rank_suit="4S")
		b = Card(rank_suit="5D")
		c = Card(rank_suit="JH")
		d = Card(rank_suit="6H")
		e = Card(rank_suit="10H")
		f = Card(rank_suit="7C") 
		g = Card(rank_suit="8C")
		hand = [a,b,c,d,e,f,g]
		hand = CombinedHand(hand)
		hand.get_hand_strength()
		self.assertEqual(hand.hand_strength, 6)

		# Same Straight
		hand2 = hand
		evaluator = Evaluator()
		evaluator.add_hands(hand, hand2)
		self.assertEqual(evaluator.get_winner(), [0,1])

		evaluator.clear_hands()


	def test_three_of_a_kind(self):
		a = Card(rank_suit="4S")
		b = Card(rank_suit="4D")
		c = Card(rank_suit="4H")
		d = Card(rank_suit="6H")
		e = Card(rank_suit="10H")
		f = Card(rank_suit="7C") 
		g = Card(rank_suit="8C")
		hand = [a,b,c,d,e,f,g]
		evaluator = CombinedHand(hand)
		evaluator.get_hand_strength()
		self.assertEqual(evaluator.hand_strength, 7)
	
	def test_three_of_a_kind(self):
		a = Card(rank_suit="4S")
		b = Card(rank_suit="4D")
		c = Card(rank_suit="4H")
		d = Card(rank_suit="6H")
		e = Card(rank_suit="10H")
		f = Card(rank_suit="7C") 
		g = Card(rank_suit="8C")
		hand = [a,b,c,d,e,f,g]
		evaluator = CombinedHand(hand)
		hh = evaluator.get_hand_strength(True)
		self.assertEqual(evaluator.hand_strength, 7)

	def test_two_pair(self):
		a = Card(rank_suit="4S")
		b = Card(rank_suit="4D")
		c = Card(rank_suit="6H")
		d = Card(rank_suit="6C")
		e = Card(rank_suit="10H")
		f = Card(rank_suit="7C") 
		g = Card(rank_suit="8C")
		hand = [a,b,c,d,e,f,g]
		evaluator = CombinedHand(hand)
		hh = evaluator.get_hand_strength()
		self.assertEqual(evaluator.hand_strength, 8)

	def test_pair(self):
		a = Card(rank_suit="4S")
		b = Card(rank_suit="4D")
		c = Card(rank_suit="KH")
		d = Card(rank_suit="6H")
		e = Card(rank_suit="10H")
		f = Card(rank_suit="7C") 
		g = Card(rank_suit="8C")
		hand = [a,b,c,d,e,f,g]
		evaluator = CombinedHand(hand)
		hh = evaluator.get_hand_strength()
		self.assertEqual(evaluator.hand_strength, 9)

	def test_high_card(self):
		a = Card(rank_suit="4S")
		b = Card(rank_suit="KD")
		c = Card(rank_suit="2H")
		d = Card(rank_suit="6H")
		e = Card(rank_suit="10H")
		f = Card(rank_suit="7C") 
		g = Card(rank_suit="8C")
		hand = [a,b,c,d,e,f,g]
		combined_hand = CombinedHand(hand)
		hh = combined_hand.get_hand_strength()
		self.assertEqual(combined_hand.hand_strength, 10)
	

# class IntegrationTests(unittest.TestCase):

# 	def test_environment(self):
# 		env = PokerEnvironment()
# 		# Add 2 Players
# 		env.add_player()
# 		env.add_player()
		
# 		env.start_new_round()
		

# To Check how fast my poker hand evaluator is
# class PerformanceTests():

if __name__ == "__main__":
	unittest.main()