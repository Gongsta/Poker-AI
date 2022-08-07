from environment import *
from evaluator import *

# Use this main.py to test some of the functionalities
def faceoff():
	s = input("Please Input the 1st card in your hand (ex: 10H): ")
	my_hand = []
	card = Card(rank_suit=s)
	my_hand.append(card)
	s = input("Please Input the 2nd card in your hand (ex: AD): ")
	card = Card(rank_suit=s)
	my_hand.append(card)

	s = input("Please Input the 1st card in your opponent's hand (ex: 10H): ")
	opponents_hand = []
	card = Card(rank_suit=s)
	opponents_hand.append(card)
	s = input("Please Input the 2nd card in your opponent's hand (ex: 10H): ")
	card = Card(rank_suit=s)
	opponents_hand.append(card)
	
	board = []
	for i in range(1, 6):
		s = input("Please Input card {} on the board: ".format(i))
		card = Card(rank_suit=s)
		board.append(card)
		
	my_hand += board
	opponents_hand += board

	my_hand = CombinedHand(my_hand)
	opponents_hand = CombinedHand(opponents_hand)
	evaluator = Evaluator()
	evaluator.add_hands(my_hand, opponents_hand)
	
	print("You have:")
	my_hand.get_hand_strength(verbose=True)
	print("Your opponent has:")
	opponents_hand.get_hand_strength(verbose=True)
	
	winners = evaluator.get_winner()
	if (len(winners) == 2):
		print("There is tie, the pot is split between you two.")
	else:
		if (winners[0] == 0):
			print("Congrulations! You won the showdown.")
		else:
			print("You lost the showdown. Your opponent takes the pot.")
	
if __name__ == "__main__":
	faceoff()