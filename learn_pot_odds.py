"""
This is a fun python script where you can enter your cards and it can help you learn
your pot odds. Built off the monte carlo algorithms in `abstraction.py` to calculate values of hands.
"""
import sys
sys.path.append('./src')

from abstraction import calculate_equity, calculate_equity_distribution, calculate_face_up_equity, plot_equity_hist
import fast_evaluator
import argparse
import random

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Learn the pot odds given your current cards and community cards. Uses Monte-Carlo methods to compute probabilities.")
	parser.add_argument("-v", "--viz",
                    action="store_true", dest="visualize_equity_distribution", default=False,
                    help="Visualize Equity distributions.")
	args = parser.parse_args()
	visualize_equity_distribution = args.visualize_equity_distribution
	
	if visualize_equity_distribution:
		deck = fast_evaluator.Deck()
		for _ in range(5000):
			random.shuffle(deck)
			player_cards = deck[:2]
			# player_cards = ['Kc', 'Qc']
			community_cards = deck[2:5]
			# community_cards = []
			equity_hist = calculate_equity_distribution(player_cards, community_cards)
			plot_equity_hist(equity_hist, player_cards, community_cards)
		
	else:
		while True:
			player_cards = []
			while len(player_cards) != 2:
				player_cards = input("Input your cards (ex: Ac 7h): ")
				player_cards = player_cards.split(" ")

			equity = calculate_equity(player_cards, [], n=10000) # We want this to be really accurate
			print("Pre-Flop Equity: {:.2f}%".format(equity * 100))
			community_cards = input("Flop Cards: ")
			community_cards = community_cards.split(" ")
			if len(community_cards) != 3:
				continue
				
			# opponent_cards = input("(Optional) Input your guess of your opponent's cards (ex: Ac 7h): ")
			# if opponent_cards == '':
			# 	opponent_cards = []
			# else:
			# 	opponent_cards = opponent_cards.split(" ")

			# equity_hist = calculate_equity_distribution(player_cards, community_cards)
			# plot_equity_hist(equity_hist)
			equity = calculate_equity(player_cards, community_cards, n=10000) # We want this to be really accurate
			print("Flop Equity: {:.2f}%".format(equity * 100))

			turn_card = input("Turn Card: ")
			if turn_card == "e":
				continue
			else:
				community_cards.append(turn_card)
			equity = calculate_equity(player_cards, community_cards, n=10000) # We want this to be really accurate
			print("Turn Equity: {:.2f}%".format(equity * 100))

			river_card = input("River Card: ")
			if turn_card == "e":
				continue
			else:
				community_cards.append(river_card)
			equity = calculate_equity(player_cards, community_cards, n=10000) # We want this to be really accurate
			print("River Equity: {:.2f}%".format(equity * 100))

			# if len(opponent_cards) == 2:
			# 	player_probability, opponent_probability = calculate_face_up_equity(player_cards, opponent_cards, community_cards, n=10000)
			# 	print("\nAssuming you know your opponent's cards, You have a {:.2f}% equity".format(player_probability * 100))
			# 	print("Your opponent has a {:.2f}% equity\n".format(opponent_probability * 100))
			
			print("\n")
			

			
