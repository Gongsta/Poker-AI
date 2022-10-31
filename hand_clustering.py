"""
This Python file calculates the equity of a given hand / paired with a board, the EHS of a pair of cards 
at different stages of the game. This assumes a random uniform draw of opponent hands and random uniform rollout of public cards.

It uses a simple Monte-Carlo technique, which samples lots of hands. Over time, it should converge 
to the expected hand strength.

We can cluster hands based on their strengths. 


While I wrote my own Card and Deck object implements, it is simply too slow. Rather, working with string representation is much faster and memory-efficient.

See this paper: https://www.cs.cmu.edu/~sandholm/potential-aware_imperfect-recall.aaai14.pdf
"""
from copy import copy, deepcopy
from typing import List
import fast_evaluator
from phevaluator import evaluate_cards
import random
import matplotlib.pyplot as plt

def calculate_equity(player_cards: List[str], community_cards=[], n=5000):
	score = 0

	initial_deck = fast_evaluator.Deck()
	for card in player_cards:
		initial_deck.remove(card)
	for card in community_cards:
		initial_deck.remove(card)
	for _ in range(n):
		deck = deepcopy(initial_deck)
		random.shuffle(deck)
		opponent_cards = deck[:2]
		final_community_cards = deepcopy(community_cards)
		while len(final_community_cards) < 5:
			final_community_cards.append(deck.pop())
		
		player_score = evaluate_cards(*(final_community_cards + player_cards))
		opponent_score = evaluate_cards(*(final_community_cards + opponent_cards))
		if player_score < opponent_score:
			score += 1
		elif player_score == opponent_score:
			score += 0.5
			
	return score / n
	
def calculate_equity_distribution(player_cards: List[str], community_cards=[], buckets=10, n=10):
	"""
	buckets = # of numbers for the histrogram
	
	The equity distribution is a better way to represent the strength of a given hand. It represents
	how well a given hand performs over various profiles of community cards. We can calculate
	the equity distribution of a hand at the following game stages: flop (we are given no community cards), turn (given 3 community cards) and river (given 4 community cards).
	
	if we want to generate a distribution for the EHS of the turn (so we are given our private cards + 3 community cards), 
	we draw various turn cards, and calculate the equity using those turn cards.  
	If we find for a given turn card that its equity is 0.645, and we have 10 buckets, we would increment the bucket 0.60-0.70 by one. 
	We repeat this process until we get enough turn card samples.

	"""
	equity_hist = [0 for _ in range(buckets)] # histogram of equities, where equity[i] represents the probability of the i-th bucket
	
	assert(len(community_cards) != 1 and len(community_cards) != 2)

	initial_deck = fast_evaluator.Deck()
	for card in player_cards:
		initial_deck.remove(card)
	for card in community_cards:
		initial_deck.remove(card)

	for _ in range(n):
		deck = deepcopy(initial_deck)
		random.shuffle(deck)
		opponent_cards = deck[:2]
		bucket_community_cards = deepcopy(community_cards)
		if len(community_cards) == 0: # We are calculating Flop Equity
			bucket_community_cards.extend([deck.pop(), deck.pop(), deck.pop()])
		else: # Turn / River Equity
			bucket_community_cards.append(deck.pop())

		score = calculate_equity(player_cards, bucket_community_cards)
		# Binary search the closest bucket value

		equity_hist[int(score * buckets)] += 1.0
	
	# Normalize the equity so that the p.m.f. of the distribution sums to 1
	for i in range(buckets):
		equity_hist[i] /= n
	
	return equity_hist
	
def plot_equity_hist(equity_hist):
	plt.hist([i/len(equity_hist) for i in range(len(equity_hist))],[i/len(equity_hist) for i in range(len(equity_hist)+1)], weights=a)
	plt.ylabel("Probability Mass")
	plt.xlabel("Equity Interval")
	plt.title("Equity Distribution")
	plt.show()
	
	
def approximate_EMD(xn, m, sorted_distances, ordered_clusters):
	"""Algorithm for efficiently approximating EMD, from 10.1609/aaai.v28i1.8816
	
	Input
	xn: Point xn with N elements
	m: mean with Q elements
	sorted_distances:
	or
	
	Output: the approximate EMD
	"""
	N = len(xn)
	Q = len(m)
	targets = [1/len(xn) for _ in range(N)]
	mean_remaining = deepcopy.copy(m)
	done = [False for _ in range(N)]
	total_cost = 0
	for i in range(Q):
		for j in range(N):
			if done[j]:
				continue
			point_cluster = xn[j]
			mean_cluster = ordered_clusters[point_cluster][i]
			amount_remaining = mean_remaining[mean_cluster]
			if amount_remaining == 0:
				continue
			
			d = sorted_distances[point_cluster][i]
			if amount_remaining < targets[j]:
				total_cost += amount_remaining * d
				targets[j] -= amount_remaining
				mean_remaining[mean_cluster] = 0
			else:
				total_cost += targets[j] * d
				targets[j] = 0
				mean_remaining[mean_cluster] -= targets[j]
				done[j] = True
	return total_cost
		
	

	

if __name__ == "__main__":
	equity_hist = calculate_equity_distribution(["Kc", "Qc"], ['Ks'])
	plot_equity_hist(equity_hist)
	# plt.plot(equity)
	# plt.show()

	

		
