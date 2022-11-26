"""
This is a fun python script where you can enter your cards and it can help you learn
your pot odds. Built off the card abstraction algorithms in `hand_clustering.py`.
"""
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
		player_cards = input("Input your cards (ex: Ac 7h): ")
		player_cards = player_cards.split(" ")
		assert(len(player_cards) == 2)
		community_cards = input("Input the cards on the board (ex: 5h Ah 3c). If none, press enter: ")
		if community_cards == '':
			community_cards = []
		else:
			community_cards = community_cards.split(" ")

		opponent_cards = input("(Optional) Input your guess of your opponent's cards (ex: Ac 7h): ")
		if opponent_cards == '':
			opponent_cards = []
		else:
			opponent_cards = opponent_cards.split(" ")


		# equity_hist = calculate_equity_distribution(player_cards, community_cards)
		# plot_equity_hist(equity_hist)
		equity = calculate_equity(player_cards, community_cards, n=10000) # We want this to be really accurate
		print("If you don't know your opponents hands, you have a {:.2f}% equity (probability of winning + 1/2 probability of chopping)".format(equity * 100))

		if len(opponent_cards) == 2:
			player_probability, opponent_probability = calculate_face_up_equity(player_cards, opponent_cards, community_cards, n=10000)
			print("\nAssuming you know your opponent's cards, You have a {:.2f}% equity".format(player_probability * 100))
			print("Your opponent has a {:.2f}% equity\n".format(opponent_probability * 100))
		

		
