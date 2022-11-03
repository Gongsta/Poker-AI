"""
This is a fun python script where you can enter your cards and it can help you learn
your pot odds. Built off the card abstraction algorithms in `hand_clustering.py`.
"""
from abstraction import calculate_equity, calculate_equity_distribution, calculate_face_up_equity

if __name__ == "__main__":
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
	equity = calculate_equity(player_cards, community_cards)
	print("If you don't know your opponents hands, you have a {:.2f}% probability of winning".format(equity * 100))

	if len(opponent_cards) == 2:
		player_probability, opponent_probability = calculate_face_up_equity(player_cards, opponent_cards, community_cards)
		print("\nAssuming you know your opponent's cards, You have a {:.2f}% probability of winning".format(player_probability * 100))
		print("Your opponent has a {:.2f}% probability of winning\n".format(opponent_probability * 100))
	

		