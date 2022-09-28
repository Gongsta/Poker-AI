import unittest
import os
import sys
import shutil
import treys
from tqdm import tqdm


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
		hand2 = [a,b,h,d,e,f,g]
		hand3 = [a,b,c,d,e,f,g]
		hand1 = CombinedHand(hand1)
		hand2 = CombinedHand(hand2)
		hand3 = CombinedHand(hand3)
		hand1.get_hand_strength()
		hand2.get_hand_strength()
		self.assertEqual(hand1.hand_strength, 2)
		self.assertEqual(hand2.hand_strength, 2)
		self.assertEqual(hand1.comparator, [4]) 
		self.assertEqual(hand2.comparator, [5]) 

		# No tie, since hand2 has a better straight flush
		evaluator = Evaluator()
		evaluator.add_hands(hand1)
		evaluator.add_hands(hand2)
		self.assertEqual(evaluator.get_winner(), [1])
		
		# Tie, since both hand1 and hand2 have the same straight flush
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
		self.assertEqual(hand1.comparator, [6, 11]) # Four cards of six + Jack Kicker
	
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
		self.assertEqual(hand1.comparator, [4, 2]) # Three cards of 4s + Pair of Twos
		
		
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
		self.assertEqual(hand1.comparator, [14,11,8,7,4])
		
		b = Card(rank_suit="3S") # Worse flush
		hand2 = [a,b,c,d,e,f,g]
		hand2 = CombinedHand(hand2)
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
		h = Card(rank_suit="9C")
		hand = [a,b,c,d,e,f,g]
		hand = CombinedHand(hand)
		hand.get_hand_strength()
		self.assertEqual(hand.hand_strength, 6)

		# Same Straight
		hand2 = hand
		evaluator = Evaluator()
		evaluator.add_hands(hand, hand2)
		self.assertEqual(evaluator.get_winner(), [0,1])

		# Better Straight
		evaluator.clear_hands()
		hand3 = [a,b,c,d,e,f,g,h]
		hand3 = CombinedHand(hand3)
		evaluator.add_hands(hand, hand3)
		self.assertEqual(evaluator.get_winner(), [1])


	def test_three_of_a_kind(self):
		a = Card(rank_suit="10S")
		b = Card(rank_suit="10H")
		c = Card(rank_suit="10D")
		d = Card(rank_suit="6H")
		e = Card(rank_suit="AH")
		f = Card(rank_suit="7C") 
		g = Card(rank_suit="8C")
		hand = [a,b,c,d,e,f,g]
		hand = CombinedHand(hand)
		hand.get_hand_strength()
		self.assertEqual(hand.hand_strength, 7)
		self.assertEqual(hand.comparator, [10,14,8])

		a = Card(rank_suit="4S")
		b = Card(rank_suit="4D")
		c = Card(rank_suit="4H")
		d = Card(rank_suit="6H")
		e = Card(rank_suit="10H")
		f = Card(rank_suit="7C") 
		g = Card(rank_suit="8C")
		hand = [a,b,c,d,e,f,g]
		hand = CombinedHand(hand)
		hand.get_hand_strength()
		self.assertEqual(hand.hand_strength, 7)
		self.assertEqual(hand.comparator, [4,10,8])
		
		evaluator = Evaluator()
		# Case #1: Better pair of threes
		a = Card(rank_suit="5S")
		b = Card(rank_suit="5D")
		c = Card(rank_suit="5H")
		hand2 = [a,b,c,d,e,f,g]
		hand2 = CombinedHand(hand2)
		
		evaluator.add_hands(hand, hand2)
		self.assertEqual(evaluator.get_winner(), [1])
		
		# Case #2 Same pair of threes, better kicker1 (Ace vs 10)
		evaluator.clear_hands()
		e = Card(rank_suit="AC")
		hand3 = [a,b,c,d,e,f,g]
		hand3 = CombinedHand(hand3)
		evaluator.add_hands(hand2, hand3)
		self.assertEqual(evaluator.get_winner(), [1])
		
		# Case #3: Same pair of threes, same kicker1, better kicker2
		evaluator.clear_hands()
		g = Card(rank_suit="9C")
		hand2 = [a,b,c,d,e,f,g]
		hand2 = CombinedHand(hand2)
		evaluator.add_hands(hand2, hand3)
		self.assertEqual(evaluator.get_winner(), [0])

		# Case #3: Same everything, time
		f = Card(rank_suit="2D")
		hand = [a,b,c,d,e,f,g]
		hand = CombinedHand(hand)
		evaluator.clear_hands()
		evaluator.add_hands(hand, hand2)
		self.assertEqual(evaluator.get_winner(), [0,1])
		
	def test_two_pair(self):
		a = Card(rank_suit="4S")
		b = Card(rank_suit="4D")
		c = Card(rank_suit="6H")
		d = Card(rank_suit="6C")
		e = Card(rank_suit="10H")
		f = Card(rank_suit="7C") 
		g = Card(rank_suit="8C")
		hand = [a,b,c,d,e,f,g]
		hand = CombinedHand(hand)
		hand.get_hand_strength()
		self.assertEqual(hand.hand_strength, 8)
		self.assertEqual(hand.comparator, [6,4,10])
		
		# Case #1: Better Top Pair
		evaluator = Evaluator()
		a = Card(rank_suit="KS")
		b = Card(rank_suit="KD")
		hand2 = [a,b,c,d,e,f,g]
		hand2 = CombinedHand(hand2)
		hand2.get_hand_strength()
		self.assertEqual(hand2.comparator, [13, 6, 10])
		evaluator.add_hands(hand, hand2)
		self.assertEqual(evaluator.get_winner(), [1])
		
		# Case #2: Better Kicker
		e = Card(rank_suit="JH")
		hand3 = [a,b,c,d,e,f,g]
		hand3 = CombinedHand(hand3)
		hand3.get_hand_strength()
		self.assertEqual(hand3.comparator, [13, 6, 11])
		evaluator.clear_hands()
		evaluator.add_hands(hand2, hand3)
		self.assertEqual(evaluator.get_winner(), [1])


	def test_pair(self):
		a = Card(rank_suit="4S")
		b = Card(rank_suit="4D")
		c = Card(rank_suit="KH")
		d = Card(rank_suit="6H")
		e = Card(rank_suit="10H")
		f = Card(rank_suit="7C") 
		g = Card(rank_suit="8C")
		hand = [a,b,c,d,e,f,g]
		hand = CombinedHand(hand)
		hand.get_hand_strength()
		self.assertEqual(hand.hand_strength, 9)
		self.assertEqual(hand.comparator, [4,13,10,8])
		
		# Case #1: Better Top Pair
		a = Card(rank_suit="QS")
		b = Card(rank_suit="QD")
		hand2 = [a,b,c,d,e,f,g]
		hand2 = CombinedHand(hand2)
		hand2.get_hand_strength()
		self.assertEqual(hand2.comparator, [12,13,10,8])

		evaluator = Evaluator()
		evaluator.add_hands(hand, hand2)
		self.assertEqual(evaluator.get_winner(), [1])
		
		# Case #2: Better Kicker #2
		e = Card(rank_suit="JH")
		hand3 = [a,b,c,d,e,f,g]
		hand3 = CombinedHand(hand3)
		hand3.get_hand_strength()
		self.assertEqual(hand3.comparator, [12,13,11,8])

		evaluator.clear_hands()
		evaluator.add_hands(hand3, hand2)
		self.assertEqual(evaluator.get_winner(), [0])
		

	def test_high_card(self):
		a = Card(rank_suit="4S")
		b = Card(rank_suit="KD")
		c = Card(rank_suit="2H")
		d = Card(rank_suit="6H")
		e = Card(rank_suit="10H")
		f = Card(rank_suit="7C") 
		g = Card(rank_suit="8C")
		hand = [a,b,c,d,e,f,g]
		hand = CombinedHand(hand)
		hand.get_hand_strength()
		self.assertEqual(hand.hand_strength, 10)
		self.assertEqual(hand.comparator, [13,10,8,7,6])

		b = Card(rank_suit="AD")
		hand2 = [a,b,c,d,e,f,g]
		hand2 = CombinedHand(hand2)
		hand2.get_hand_strength()
		self.assertEqual(hand2.hand_strength, 10)
		self.assertEqual(hand2.comparator, [14,10,8,7,6])
		
		# Case #1: Better High Card
		evaluator = Evaluator()
		evaluator.add_hands(hand, hand2)
		self.assertEqual(evaluator.get_winner(), [1])
		
		# Case #2: Better Second High Card
		e = Card(rank_suit="JH")
		hand = [a,b,c,d,e,f,g]
		hand = CombinedHand(hand)
		evaluator.clear_hands()
		evaluator.add_hands(hand, hand2)
		self.assertEqual(evaluator.get_winner(), [0])
	
		# Case #3: Tie
		hand2 = hand
		evaluator.clear_hands()
		evaluator.add_hands(hand, hand2)
		self.assertEqual(evaluator.get_winner(), [0,1])

	def test_with_treys(self):
		# Use the treys library to test that our hand evaluator is always correct
		deck = Deck()
		treys_evaluator = treys.Evaluator()
		for i in tqdm(range(250000)):
			deck.reset_deck()
			board = []
			treys_board = []
			player = []
			treys_player = []
			opponent = []
			treys_opponent = []
			for _ in range(2):
				card = deck.draw()
				player.append(card)
				treys_player.append(treys.Card.new(str(card)))
			
			for _ in range(2):
				card = deck.draw()
				opponent.append(card)
				treys_opponent.append(treys.Card.new(str(card)))

			for _ in range(5):
				card = deck.draw()
				board.append(card)
				treys_board.append(treys.Card.new(str(card)))
			
			evaluator = Evaluator()
			evaluator.add_hands(CombinedHand(player+board), CombinedHand(opponent+board))
			p1_score = treys_evaluator.evaluate(treys_board, treys_player)
			p2_score = treys_evaluator.evaluate(treys_board, treys_opponent)
			if p1_score < p2_score:
				treys_winner = [0]
			elif p1_score > p2_score:
				treys_winner = [1]
			else:
				treys_winner = [0,1]

			if (evaluator.get_winner() != treys_winner): # In the case of errors
				for hand in evaluator.hands:
					print(hand.comparator)
				
				print(evaluator)

			self.assertEqual(evaluator.get_winner(), treys_winner)
		
class IntegrationTests(unittest.TestCase):

	def test_environment(self):
		env = PokerEnvironment()
		# Add 2 Players
		env.add_player()
		env.add_player()
		
		env.start_new_round()
		

# To Check how fast my poker hand evaluator is
# class PerformanceTests():

if __name__ == "__main__":
	unittest.main()