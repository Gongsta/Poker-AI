"""
Python file that takes care of betting and card abstractions for Poker. 

Inspired from Noam Brown's paper: https://arxiv.org/pdf/1805.08195.pdf

"The blueprint abstraction treats every poker hand separately on the first betting round (where there are
169 strategically distinct hands). On the remaining betting rounds, the hands are grouped into 30,000
buckets. The hands in each bucket are treated identically and have a shared strategy, so
they can be thought as sharing an abstract infoset".

*Note on Card Abstraction: While I wrote my own Card and Deck object implements, it is simply too slow. Rather, working with string representation is much faster and memory-efficient.

"""

# Preflop Abstraction
"""
For the Pre-flop, we can make a lossless abstraction with exactly 169 buckets. The idea here is that what specific suits
our private cards are doesn't matter. The only thing that matters is whether both cards are suited or not.

This is how the number 169 is calculated:
- For cards that are not pocket pairs, we have (13 choose 2) = 13 * 12 / 2 = 78 buckets (since order doesn't matter)
- These cards that are not pocket pairs can also be suited, so we must differentiate them. We have 78 * 2 = 156 buckets
- Finally, for cards that are pocket pairs, we have 13 extra buckets (Pair of Aces, Pair of 2, ... Pair Kings). 156 + 13 = 169 buckets

Note that a pair cannot be suited, so we don't need to multiply by 2.
"""


# Flop/Turn/River Abstraction

"""
We the equity of a given hand / paired with a board, the EHS of a pair of cards 
at different stages of the game, which is calculated assuming a random uniform draw of opponent hands and random uniform rollout of public cards.

It uses a simple Monte-Carlo method, which samples lots of hands. Over lots of iterations, it will converge 
to the expected hand strength.

We can cluster hands using K-Means++ to cluster hands of similar distance.


See this paper: https://www.cs.cmu.edu/~sandholm/potential-aware_imperfect-recall.aaai14.pdf
"""
from copy import copy, deepcopy
from typing import List
import fast_evaluator
from phevaluator import evaluate_cards
import random
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import json
import time

def calculate_equity(player_cards: List[str], community_cards=[], n=100, timer=False):
	if timer:
		start_time = time.time()
	wins = 0
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
			wins += 1
		elif player_score == opponent_score:
			wins += 0.5
			
	if timer:
		print("Time it takes to call function: {}s".format(time.time() - start_time))

	return wins / n
	
def calculate_face_up_equity(player_cards, opponent_cards, community_cards, n=1000):
	"""Same as calculate_equity, except that you know your opponents cards as well, so total probability should sum to one.
	"""
	assert(len(player_cards) == 2 and len(opponent_cards) == 2)
	player_wins = 0
	opponent_wins = 0

	initial_deck = fast_evaluator.Deck()
	for card in player_cards:
		initial_deck.remove(card)
	for card in opponent_cards:
		initial_deck.remove(card)
	for card in community_cards:
		initial_deck.remove(card)
	for _ in range(n):
		deck = deepcopy(initial_deck)
		random.shuffle(deck)
		final_community_cards = deepcopy(community_cards)
		while len(final_community_cards) < 5:
			final_community_cards.append(deck.pop())
		
		player_score = evaluate_cards(*(final_community_cards + player_cards))
		opponent_score = evaluate_cards(*(final_community_cards + opponent_cards))
		if player_score < opponent_score:
			player_wins += 1
		elif player_score == opponent_score:
			player_wins += 0.5
			opponent_wins += 0.5
		else:
			opponent_wins += 1

			
	return player_wins / n, opponent_wins / n

def calculate_equity_distribution(player_cards: List[str], community_cards=[], buckets=5, n=200, timer=False):
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
	if timer:
		start_time = time.time()
	equity_hist = [0 for _ in range(buckets)] # histogram of equities, where equity[i] represents the probability of the i-th bucket
	
	assert(len(community_cards) != 1 and len(community_cards) != 2)

	initial_deck = fast_evaluator.Deck()
	for card in player_cards:
		initial_deck.remove(card)
	for card in community_cards:
		initial_deck.remove(card)

	for i in range(n):
		deck = deepcopy(initial_deck)
		random.shuffle(deck)
		bucket_community_cards = deepcopy(community_cards)
		if len(community_cards) == 0: # We are calculating Flop Equity
			bucket_community_cards.extend([deck.pop(), deck.pop(), deck.pop()])
		elif len(community_cards) < 5: # Turn / River Equity
			bucket_community_cards.append(deck.pop())

		score = calculate_equity(player_cards, bucket_community_cards)

		equity_hist[min(int(score * buckets), buckets-1)] += 1.0 # Score of the closest bucket is incremented by 1
	
	# Normalize the equity so that the probability mass function (p.m.f.) of the distribution sums to 1
	for i in range(buckets):
		equity_hist[i] /= n
	
	if timer:
		print("Time to calculate equity distribution: ", time.time() - start_time)
	return equity_hist
	
def plot_equity_hist(equity_hist, player_cards=None, community_cards=None):
	"""Plot the equity histogram.
	"""
	plt.clf() # Clear Canvas
	plt.hist([i/len(equity_hist) for i in range(len(equity_hist))],[i/len(equity_hist) for i in range(len(equity_hist)+1)], weights=equity_hist)
	plt.ylabel("Probability Mass")
	plt.xlabel("Equity Interval")
	if player_cards:
		player_string = "\nPlayer Cards: " + str(player_cards)
	else:
		player_string = ""
	
	if community_cards:
		community_string = "\nCommunity Cards: " + str(community_cards)
	else:
		community_string = ""

	plt.title("Equity Distribution" + player_string + community_string)
	plt.show(block=False) # to plot graphs consecutively quickly with non-blocking behavior
	plt.pause(0.2)
	
	
"""
The Algorithm: 
1. For river, generate 5000 clusters, each representing a particular equity distribution.
2. For turn, generate 5000 clusters which are potential-aware, meaning it takes into consideration
the probability of transitioning into the next clusters (of the river).

So if you draw the river card, what is the probability you end up in a given river cluster. The river clusters are non 
overlapping, so you always end up in one.

3. For flop, " " (of the turn)
4. For pre-flop, generate 169 clusters (lossless abstraction).

"""

def cluster_hands(n_clusters: int, total_community_cards=3):
	kmeans = KMeans(n_clusters=n_clusters)
	initial_deck = fast_evaluator.Deck()
	X = []
	
	for _ in range(5000):
		deck = random.shuffle(initial_deck)
		
		player_cards = deck[:2]
		community_cards = deck[2:2+total_community_cards]
		
		X.append(calculate_equity_distribution(player_cards, community_cards))

		
def generate_preflop_clusters(): # Lossless abstraction for pre-flop, 169 buckets
	preflop_clusters = {}
	cluster_i = 0
	for rank in ["A", "2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K"]:
		preflop_clusters[cluster_i] = rank + 's' # Suited
		cluster_i += 1

	for rank in ["A", "2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K"]:
		preflop_clusters[cluster_i] = rank + 'o' # Offsuit
		cluster_i += 1

	for rank in ["A", "2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K"]:
		preflop_clusters[cluster_i] = 2 * rank  # Pairs
		cluster_i += 1
	
	assert(cluster_i == 169) # There should be exactly 169 buckets

	return preflop_clusters


def approximate_EMD(xn, m, sorted_distances, ordered_clusters):
	"""Algorithm for efficiently approximating EMD, from 10.1609/aaai.v28i1.8816. Don't know if I am actually going to use this.
	
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
		
def visualize_clustering():
	# TODO: Visualize the clustering by actually seeing how the distributions shown
	return

if __name__ == "__main__":
	opponent_cards = []
	
	deck = fast_evaluator.Deck()

	random.shuffle(deck)
	player_cards = deck[:2]
	community_cards = deck[2:5]

	for _ in range(5000):
		equity_hist = calculate_equity_distribution(player_cards, community_cards)
		plot_equity_hist(equity_hist, player_cards, community_cards)
	