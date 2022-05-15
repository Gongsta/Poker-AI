import unittest
import os
import sys
import shutil


if __name__ == "__main__":
	devpath = os.path.relpath(os.path.join('..'), start=os.path.dirname(__file__))
	sys.path = [devpath] + sys.path

# Import Libraries
from environment import *

class TestCards(unittest.TestCase):
	
	def test_card_initialization(self):
		jackOfHearts = Card(11, "Hearts")
		self.assertEqual(jackOfHearts.rank, 11)
		self.assertEqual(jackOfHearts.suit, "Hearts")
		
		with self.assertRaises(Exception):
			Card(13, "Diamonds")
		with self.assertRaises(Exception):
			Card(1, "B")

	def test_deck_initalization(self):
		new_deck = Deck()
		

if __name__ == "__main__":
	unittest.main()