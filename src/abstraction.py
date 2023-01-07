"""
Python file that takes care of betting and card abstractions for Poker. 

Inspired from Noam Brown's paper: https://arxiv.org/pdf/1805.08195.pdf

"The blueprint abstraction treats every poker hand separately on the first betting round (where there are
169 strategically distinct hands). On the remaining betting rounds, the hands are grouped into 30,000
buckets. The hands in each bucket are treated identically and have a shared strategy, so
they can be thought as sharing an abstract infoset".

*Note on Card Abstraction: While I wrote my own Card and Deck object implements, it is simply too slow. Rather, working with string representation is much faster and memory-efficient.

"""
from copy import deepcopy
from typing import List
import fast_evaluator
from phevaluator import evaluate_cards
import random
import matplotlib.pyplot as plt
import time
import numpy as np
import os
import glob
import joblib
from joblib import Parallel, delayed
import torch # To run K-means with GPU
import ot

"""
BET ABSTRACTION
"""
# For action abstraction, I have decided to simplify the actions to fold (f), check (k), call (c), small-bet (0.5x pot), medium-bet (1x pot), large-bet (2x pot), and all-in. 

# def bet_abstraction(bet_size):
# 	"""Bet size is relative to pot.
	
# 	"""
# 	if bet_size == 0:
# 		return 'c'
# 	elif bet_size <= 0.5:
# 		return 0.5
# 	elif bet_size <= 1.0:
# 		return 1.0
# 	elif bet_size <= 2.0:
# 		return 2.0
# 	else:
# 		return 'all-in'

# def abstraction():
# 			# TODO: Investigate the effect of action abstraction on exploitability.
# 			"""
			
# 			Daniel Negreanu: How Much Should You Raise? https://www.youtube.com/watch?v=WqRUyYQcc5U
# 			Bet sizing: https://www.consciouspoker.com/blog/poker-bet-sizing-strategy/#:~:text=We%20recommend%20using%201.5x,t%20deduce%20your%20likely%20holdings.
# 			Also see slumbot notes: https://nanopdf.com/queue/slumbot-nl-solving-large-games-with-counterfactual_pdf?queue_id=-1&x=1670505293&z=OTkuMjA5LjUyLjEzOA==
			
# 			TODO: Check the case on preflop when the small blind simply calls, the BB should have the option to min-raise by amounts.


# 			For initial bets, these are fractions of the total pot size (money at the center of the table):
# 			for bets:
# 				- b0.25 = bet 25% of the pot 
# 				- b0.5 = bet 50% of the pot
# 				- b0.75 = bet 75% of the pot
# 				- b1 = bet 100% of the pot
# 				- b2 = ...
# 				- b4 = ...
# 				- b8 =
# 				- all-in = all-in, opponent is forced to either call or fold

# 			After a bet has happened, we can only raise by a certain amount.
# 				- b0.5
# 				- b1 
# 				- b2 = 2x pot size
# 				- b4 = 4x pot size
# 				- b8 = 8x pot size
# 				- all-in = all-in, opponent is forced to either call or fold
			
# 			2-bet is the last time we can raise again
# 			- b1
# 			- b2 = 2x pot size
# 			- all-in
			
# 			3-bet
# 			- b1
			
# 			4-bet
# 			- all-in
# 			"""
			
# 			# Note: all-in case is just the maximum bet

# 		actions = ['k', 'b0.25','b0.5', 'b0.75', 'b1', 'b2', 'b4', 'b8', 'all-in', 'c', 'f'] 
		
# 		current_game_stage_history, stage = self.get_current_game_stage_history()

# 		# Pre-flop
# 		if stage == 'preflop':
# 		# Small blind to act
# 			if len(current_game_stage_history) == 0: # call/bet
# 				actions.remove('k') # You cannot check
# 				return actions

# 			# big blind to act
# 			elif len(current_game_stage_history) == 1: # 2-bet
# 				if (current_game_stage_history[0] == 'c'): # Small blind called, you don't need to fold
# 					actions.remove('f')
# 					return actions
# 				else: # Other player has bet, so you cannot check
# 					actions.remove('k')
# 					return actions
# 			elif len(current_game_stage_history) == 2: # 3-bet
# 				# You cannot check at this point
# 				actions = ['b1', 'all-in', 'c', 'f']
				
# 			elif len(current_game_stage_history) == 3: # 4-bet
# 				actions = ['all-in', 'c', 'f'] 

# 		else: # flop, turn, river
# 			if len(current_game_stage_history == 0):
# 				actions.remove('f') # You cannot fold
# 			elif len(current_game_stage_history) == 1:
# 				if current_game_stage_history[0] == 'k':
# 					actions.remove('f')
# 				else: # Opponent has bet, so you cannot check
# 					actions.remove('k')

# 		return actions
# 	else:
# 		raise Exception("Cannot call actions on a terminal history")


"""
CARD ABSTRACTION
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
to the expected hand strength. To have a descriptive description of the potential of a hand, I use
an equity distribution rather than a scalar value. This idea was taken from this paper: https://www.cs.cmu.edu/~sandholm/potential-aware_imperfect-recall.aaai14.pdf

This kind of abstraction is used by all superhuman Poker AIs.

We can cluster hands using K-Means to cluster hands of similar distance. The distance metric used is Earth Mover's 
Distance, which is taken from the Python Optiaml Transport Library.

How do I find the optimal number of clusters?


"""
from functools import partial
import numpy as np
import torch
from tqdm import tqdm


# Modified version of K-Means to add Earth Mover's Distance from here: https://github.com/subhadarship/kmeans_pytorch/blob/master/kmeans_pytorch/__init__.py
def initialize(X, n_clusters, seed):
	"""
	initialize cluster centers
	:param X: (torch.tensor) matrix
	:param n_clusters: (int) number of clusters
	:param seed: (int) seed for kmeans
	:return: (np.array) initial state
	"""
	num_samples = len(X)
	if seed == None:
		indices = np.random.choice(num_samples, n_clusters, replace=False)
	else:
		np.random.seed(seed) ; indices = np.random.choice(num_samples, n_clusters, replace=False)
	initial_state = X[indices]
	return initial_state

def kmeans_custom(
		X,
		n_clusters,
		distance='euclidean',
		centroids=[],
		tol=1e-3,
		tqdm_flag=True,
		iter_limit=0,
		device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
		seed=None,
):
	"""
	perform kmeans n_init, default=10
	Number of time the k-means algorithm will be run with different centroid seeds. The final results will be the best output of n_init consecutive runs in terms of inertia.
	:param X: (torch.tensor) matrix
    :param num_clusters: (int) number of clusters
    :param distance: (str) distance [options: 'euclidean', 'cosine'] [default: 'euclidean']
    :param seed: (int) seed for kmeans
    :param tol: (float) threshold [default: 0.001]
    :param device: (torch.device) device [default: cpu]
    :param tqdm_flag: Allows to turn logs on and off
    :param iter_limit: hard limit for max number of iterations
    :param gamma_for_soft_dtw: approaches to (hard) DTW as gamma -> 0
	Return
		X_cluster_ids (torch.tensor), centroids (torch.tensor)
	"""
	if tqdm_flag:
		print(f'running k-means on {device}..')

	if distance == 'euclidean':
		pairwise_distance_function = partial(pairwise_distance, device=device, tqdm_flag=tqdm_flag)
	elif distance == 'cosine':
		pairwise_distance_function = partial(pairwise_cosine, device=device)
	elif distance == 'EMD':
		pairwise_distance_function = partial(pairwise_EMD, device=device)

	else:
		raise NotImplementedError

	if type(X) != torch.Tensor:
		X = torch.tensor(X)
	# convert to float
	X = X.float()

	# transfer to device
	X = X.to(device)

	# initialize
	if type(centroids) == list:  # ToDo: make this less annoyingly weird
		initial_state = initialize(X, n_clusters, seed=seed)
	else:
		if tqdm_flag:
			print('resuming')
		# find data point closest to the initial cluster center
		initial_state = centroids
		dis = pairwise_distance_function(X, initial_state)
		choice_points = torch.argmin(dis, dim=0)
		initial_state = X[choice_points]
		initial_state = initial_state.to(device)

	iteration = 0
	if tqdm_flag:
		tqdm_meter = tqdm(desc='[running kmeans]')

	while True:
		dis = pairwise_distance_function(X, initial_state)

		choice_cluster = torch.argmin(dis, dim=1)

		initial_state_pre = initial_state.clone()

		for index in range(n_clusters):
			selected = torch.nonzero(choice_cluster == index).squeeze().to(device)

			selected = torch.index_select(X, 0, selected)

			# https://github.com/subhadarship/kmeans_pytorch/issues/16
			if selected.shape[0] == 0:
				selected = X[torch.randint(len(X), (1,))]

			initial_state[index] = selected.mean(dim=0)

		center_shift = torch.sum(
			torch.sqrt(
				torch.sum((initial_state - initial_state_pre) ** 2, dim=1)
			))

		# increment iteration
		iteration = iteration + 1

		# update tqdm meter
		if tqdm_flag:
			tqdm_meter.set_postfix(
				iteration=f'{iteration}',
				center_shift=f'{center_shift ** 2:0.6f}',
				tol=f'{tol:0.6f}'
			)
			tqdm_meter.update()
		if center_shift ** 2 < tol:
			break
		if iter_limit != 0 and iteration >= iter_limit:
			break

	return choice_cluster.cpu(), initial_state.cpu() # clusters_indices_on_initial data, final_centroids


def kmeans_custom_predict(
		X,
		centroids,
		distance='euclidean',
		device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
		tqdm_flag=True
):
	"""
	Return
		cluster_ids_on_X (torch.tensor)
	
	"""
	if tqdm_flag:
		print(f'predicting on {device}..')

	if distance == 'euclidean':
		pairwise_distance_function = partial(pairwise_distance, device=device, tqdm_flag=tqdm_flag)
	elif distance == 'cosine':
		pairwise_distance_function = partial(pairwise_cosine, device=device)
	elif distance == 'EMD':
		pairwise_distance_function = partial(pairwise_EMD, device=device)
	else:
		raise NotImplementedError

	# convert to float
	if type(X) != torch.Tensor:
		X = torch.tensor(X)
	X = X.float()

	# transfer to device
	X = X.to(device)

	dis = pairwise_distance_function(X, centroids)
	if (len(dis.shape) == 1): # Prediction on a single data
		choice_cluster = torch.argmin(dis)
		
	else: 
		choice_cluster = torch.argmin(dis, dim=1)

	return choice_cluster.cpu()


def pairwise_distance(data1, data2, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), tqdm_flag=True):
	if tqdm_flag:
		print(f'device is :{device}')
	
	# transfer to device
	data1, data2 = data1.to(device), data2.to(device)

	A = data1.unsqueeze(dim=1) # N*1*M
	B = data2.unsqueeze(dim=0) # 1*N*M

	dis = (A - B) ** 2.0
	# return N*N matrix for pairwise distance
	dis = dis.sum(dim=-1).squeeze()
	return dis


def pairwise_cosine(data1, data2, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
	# transfer to device
	data1, data2 = data1.to(device), data2.to(device)

	A = data1.unsqueeze(dim=1) # N*1*M
	B = data2.unsqueeze(dim=0) # 1*N*M

	# normalize the points  | [0.3, 0.4] -> [0.3/sqrt(0.09 + 0.16), 0.4/sqrt(0.09 + 0.16)] = [0.3/0.5, 0.4/0.5]
	A_normalized = A / A.norm(dim=-1, keepdim=True)
	B_normalized = B / B.norm(dim=-1, keepdim=True)

	cosine = A_normalized * B_normalized

	# return N*N matrix for pairwise distance
	cosine_dis = 1 - cosine.sum(dim=-1).squeeze()
	return cosine_dis


def pairwise_EMD(data1, data2, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
	assert(len(data1.shape) == 2)
	assert(len(data2.shape) == 2)
	assert(data1.shape[1] == data2.shape[1])
	n = data1.shape[1]
	pos_a = torch.tensor([[i] for i in range(n)])
	pos_b = torch.tensor([[i] for i in range(n)])
	C = ot.dist(pos_a, pos_b, metric='euclidean')
	C.to(device)

	# Correct solution, but very slow
	dist = torch.zeros((data1.shape[0], data2.shape[0]))
	for i, hist_a in enumerate(data1):
		for j, hist_b in enumerate(data2):
			for _ in range(10):  # Janky fix for small precision error
				try:
					ot_emd = ot.emd(hist_a, hist_b, C, numThreads="max")  # The precision is set to 7, so sometimes the sum doesn't get to precisely 1. 
					break
				except Exception as e:
					print(e)
					continue
				
			transport_cost_matrix = ot_emd * C
			dist[i][j] = transport_cost_matrix.sum()
	
	return dist

def kmeans_search(X):
	"""
	We can check for the quality of clustering by checking the inter-cluster distance, using the 
	same metric that we used for EMD.
	
	At some point, there is no point in increasing the number of clusters, since we don't really
	get more information.
	"""
	# Search for the optimal number of clusters through a grid like search

	if type(X) != torch.Tensor:
		X = torch.tensor(X)
	# convert to float
	X = X.float()

	# n_clusters = [10, 25, 50, 100, 200, 1000, 5000]
	n_clusters = [5000]
	for n_cluster in n_clusters:
		cluster_indices, centroids = kmeans(X, n_cluster)
		X_cluster_centroids = centroids[cluster_indices]
		distances = 0
		for i, X_cluster_centroid in enumerate(X_cluster_centroids):
			distances += pairwise_distance(torch.unsqueeze(X_cluster_centroid, axis=0), torch.unsqueeze(X[i], axis=0), tqdm_flag=False)
		print(f"Sum of cluster to data distance {distances}")
		print(f"Mean cluster to data distance {distances / X_cluster_centroids.shape[0]}")


	



def calculate_equity(player_cards: List[str], community_cards=[], n=1000, timer=False):
	if timer:
		start_time = time.time()
	wins = 0
	deck = fast_evaluator.Deck(excluded_cards=player_cards + community_cards)
	
	for _ in range(n):
		random.shuffle(deck)
		opponent_cards = deck[:2] # To avoid creating redundant copies
		player_score = evaluate_cards(*(player_cards + community_cards + deck[2:2+(5 - len(community_cards))]))
		opponent_score = evaluate_cards(*(opponent_cards + community_cards + deck[2:2+(5 - len(community_cards))]))
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

	deck = fast_evaluator.Deck(excluded_cards=player_cards + opponent_cards + community_cards)
	for _ in range(n):
		random.shuffle(deck)
		player_score = evaluate_cards(*(player_cards + community_cards + deck[0:(5 - len(community_cards))]))
		opponent_score = evaluate_cards(*(opponent_cards + community_cards + deck[0:(5 - len(community_cards))]))

		if player_score < opponent_score:
			player_wins += 1
		elif player_score == opponent_score:
			player_wins += 0.5
			opponent_wins += 0.5
		else:
			opponent_wins += 1

			
	return player_wins / n, opponent_wins / n

def calculate_equity_distribution(player_cards: List[str], community_cards=[], bins=5, n=200, timer=False, parallel=True):
	"""
	Return
		equity_hist - Histogram as a list of "bins" elements

	n = # of cards to sample from the next round to generate this distribution.
	
	There is a tradeoff between the execution speed and variance of the values calculated, since
	we are using a monte-carlo method to calculate those equites. In the end, I found a bin=5, n=100 
	and rollouts using 100 values to be a good approximation. We won't be using this method for 
	pre-flop, since we can have a lossless abstraction of that method anyways.
	
	The equity distribution is a better way to represent the strength of a given hand. It represents
	how well a given hand performs over various profiles of community cards. We can calculate
	the equity distribution of a hand at the following game stages: flop (we are given no community cards), turn (given 3 community cards) and river (given 4 community cards).
	
	if we want to generate a distribution for the EHS of the turn (so we are given our private cards + 3 community cards), 
	we draw various turn cards, and calculate the equity using those turn cards.  
	If we find for a given turn card that its equity is 0.645, and we have 10 bins, we would increment the bin 0.60-0.70 by one. 
	We repeat this process until we get enough turn card samples.
	"""
	if timer:
		start_time = time.time()
	equity_hist = [0 for _ in range(bins)] # histogram of equities, where equity[i] represents the probability of the i-th bin
	
	assert(len(community_cards) != 1 and len(community_cards) != 2)

	deck = fast_evaluator.Deck(excluded_cards=player_cards + community_cards)

	if parallel: # Computing these equity distributions in parallel is much faster
		def sample_equity():
			random.shuffle(deck)
			if len(community_cards) == 0:
				score = calculate_equity(player_cards, community_cards + deck[:3], n=200)
			elif (len(community_cards) < 5):
				score = calculate_equity(player_cards, community_cards + deck[:1], n=100)
			else:
				score = calculate_equity(player_cards, community_cards, n=100)

			# equity_hist[min(int(score * bins), bins-1)] += 1.0 # Score of the closest bucket is incremented by 1
			return min(int(score * bins), bins-1)
		
		equity_bin_list = Parallel(n_jobs=-1)(delayed(sample_equity)() for _ in range(n))
		for bin_i in equity_bin_list:
			equity_hist[bin_i] += 1.0

	else:
		for i in range(n):
			random.shuffle(deck)
			if len(community_cards) == 0:
				score = calculate_equity(player_cards, community_cards + deck[:3], n=200)
			else:
				score = calculate_equity(player_cards, community_cards + deck[:1], n=100)

			equity_hist[min(int(score * bins), bins-1)] += 1.0 # Score of the closest bucket is incremented by 1
	
	# Normalize the equity so that the probability mass function (p.m.f.) of the distribution sums to 1
	for i in range(bins):
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
		
def generate_preflop_clusters(): # Lossless abstraction for pre-flop, 169 clusters
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

def create_abstraction_folders():
	if not os.path.exists('../data'):
		for split in ['clusters', 'raw']:
			for stage in ['flop', 'turn', 'river']:
				os.makedirs(f'../data/{split}/{stage}')


def generate_postflop_equity_distributions(n_samples, bins,stage=None, save=True, timer=True): # Lossful abstraction for flop, turn and river
	if timer:
		start_time = time.time()
	assert(stage is None or stage == 'flop' or stage == 'turn' or stage == 'river')
	equity_distributions = []
	hands = []
	
	
	if stage is None:
		generate_postflop_equity_distributions(n_samples, bins, 'flop', save, timer)
		generate_postflop_equity_distributions(n_samples, bins, 'turn', save, timer)
		generate_postflop_equity_distributions(n_samples, bins, 'river', save, timer)
	elif stage == 'flop':
		num_community_cards = 3
	elif stage == 'turn':
		num_community_cards = 4
	elif stage == 'river':
		num_community_cards = 5
	
	deck = fast_evaluator.Deck()
	for i in tqdm(range(n_samples)):
		random.shuffle(deck)
		
		player_cards = deck[:2]
		community_cards = deck[2:2+num_community_cards]
		distribution = calculate_equity_distribution(player_cards, community_cards, bins)
		equity_distributions.append(distribution)
		hands.append(' '.join(player_cards + community_cards))
		
	
	assert(len(equity_distributions) == len(hands))

	equity_distributions = np.array(equity_distributions)
	print(equity_distributions)
	if save:
		create_abstraction_folders()
		file_id = int(time.time()) # Use the time as the file_id
		# TODO: Change to joblib saving for consistency everywhere
		with open(f'../data/raw/{stage}/{file_id}_samples={n_samples}_bins={bins}.npy', 'wb') as f:
			np.save(f, equity_distributions)
		joblib.dump(hands, f'../data/raw/{stage}/{file_id}_samples={n_samples}_bins={bins}')  # Store the list of hands, so you can associate a particular distribution with a particular hand
	
	


def visualize_clustering():
	"""Visualization higher dimensional data is super interesting.
	
	"""

	# TODO: Visualize the clustering by actually seeing how the distributions shown
	return

		

def get_filenames(folder, extension='.npy'):
	filenames = []
	
	for path in glob.glob(os.path.join(folder, '*' + extension)):
		# Extract the filename
		filename = os.path.split(path)[-1]        
		filenames.append(filename)

	return filenames


	
import argparse
from sklearn.cluster import KMeans
if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Generate Poker Hand Abstractions.")
	parser.add_argument("-g", "--generate",
                    action="store_true", dest="generate", default=False,
                    help="Generate Abstractions.")
	parser.add_argument("--n_samples", default=10000, 
                    dest="n_samples",
                    help="Number of samples to sample from to generate the abstraction.")
	parser.add_argument("--n_clusters", default=50, 
                    dest="n_clusters",
                    help="Number of clusters to generate.")
	parser.add_argument("-b", "--bins", default=5, 
                    dest="bins",
                    help="The granularity of your generated data.")
	parser.add_argument("-s", "--stage", default='turn', 
                    dest="stage", 
                    help="Select the stage of the game that you would like to abstract (flop, turn, river).")
	# Hyperparamtesrs
	args = parser.parse_args()

	generate = args.generate # Generate histogram distributions to cluster on
	clustering = True # Cluster these histogram distributions

	stage = args.stage
	n_samples = int(args.n_samples)
	bins = args.bins

	if generate:
		generate_postflop_equity_distributions(n_samples, bins, stage)
	
	if clustering:
		raw_dataset_filenames = sorted(get_filenames(f'../data/raw/{stage}'))
		filename = raw_dataset_filenames[-1] # Take the most recently generated dataset to run our clustering on
		
		equity_distributions = np.load(f'../data/raw/{stage}/{filename}') # TODO: Switch to joblib
		print(filename)
		if not os.path.exists(f'../data/clusters/{stage}/{filename}'):
			print(f"Generating the cluster for the {stage}")
			print(filename)
			kmeans = KMeans(100) # 100 Clusters seems good using the Elbow Method, see notebook/abstraction_exploration.ipynb
			kmeans.fit(equity_distributions) # Perform Clustering
			centroids = kmeans.cluster_centers_
			joblib.dump(centroids, f'../data/clusters/{stage}/{filename}')
		else: # Centroids have already been generated, just load them, which are tensors
			centroids = joblib.load(f'../data/clusters/{stage}/{filename}')
			# Load KMeans Model
			kmeans = KMeans(100)
			kmeans.cluster_centers_ = centroids
			kmeans._n_threads = -1
		
		
		# # Visualization of the hands
		# hands = joblib.load(f'data/raw/{stage}/{filename.split(".")[0]}')  
		# for i in range(equity_distributions.shape[0]):
		# 	hand = hands[i]
		# 	hand = hand.split(' ')
		# 	player_cards = hand[0]
		# 	community_cards = hand[0]
		# 	plot_equity_hist(equity_distributions[i], player_cards, community_cards)
		
		# Visualize the clusstering


		

		
	
	
