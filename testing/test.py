import unittest
import os
import sys
import shutil


if __name__ == "__main__":
	devpath = os.path.relpath(os.path.join('..'), start=os.path.dirname(__file__))
	sys.path = [devpath] + sys.path

# Import Libraries
from environment import *

class UnitTests(unittest.TestCase):
	
	# Unit Testing
	def test_card_initialization(self):
		jackOfHearts = Card(11, "Hearts")
		self.assertEqual(jackOfHearts.rank, 11)
		self.assertEqual(jackOfHearts.suit, "Hearts")
		
		with self.assertRaises(Exception):
			Card(13, "Diamonds")
		with self.assertRaises(Exception):
			Card(1, "Diamonds")
		with self.assertRaises(Exception):
			Card(2, "B")

	def test_deck_initalization(self):
		new_deck = Deck()
		assert(new_deck.total_remaining_cards == 52)
		
class IntegrationTests(unittest.TestCase):

	def test_environment(self):
		env = PokerEnvironment()
		# Add 2 Players
		env.add_player()
		env.add_player()
		
		env.start_new_round()
		
	

if __name__ == "__main__":
	unittest.main()