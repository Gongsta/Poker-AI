"""
Python file that takes care of betting and card abstractions for Poker, used for training.

For BET ABSTRACTION, the logic is directly encoded into the CFR training (see `postflop_holdem.py` for an example)

CARD ABSTRACTION

Description:
We the equity of a given hand / paired with a board, the EHS of a pair of cards.
at different stages of the game, which is calculated assuming a random uniform draw of opponent hands and random uniform rollout of public cards.

It uses a simple Monte-Carlo method, which samples lots of hands. Over lots of iterations, it will converge
to the expected hand strength. To have a descriptive description of the potential of a hand, I use
an equity distribution rather than a scalar value. This idea was taken from this paper: https://www.cs.cmu.edu/~sandholm/potential-aware_imperfect-recall.aaai14.pdf

This kind of abstraction is used by all superhuman Poker AIs.

We can cluster hands using K-Means to cluster hands of similar distance. The distance metric used is Earth Mover's
Distance, which is taken from the Python Optiaml Transport Library.

How do I find the optimal number of clusters?
"""

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
from tqdm import tqdm
from fast_evaluator import phEvaluatorSetup
import argparse
from sklearn.cluster import KMeans

USE_KMEANS = False  # use kmeans if you want to cluster by equity distribution (more refined, but less accurate)
NUM_FLOP_CLUSTERS = 10
NUM_TURN_CLUSTERS = 10
NUM_RIVER_CLUSTERS = 10

if USE_KMEANS:
    # See `notebook/abstraction_exploration.ipynb` for some exploration of how many clusters to use
    NUM_FLOP_CLUSTERS = 50
    NUM_TURN_CLUSTERS = 50
    NUM_RIVER_CLUSTERS = 10  # For river, you can just compute equity
    load_kmeans_classifiers()


def evaluate_winner(board, player_hand, opponent_hand):
    p1_score = evaluate_cards(*(board + player_hand))
    p2_score = evaluate_cards(*(board + opponent_hand))
    if p1_score < p2_score:
        return 1
    elif p1_score > p2_score:
        return -1
    else:
        return 0


# ----- Load the pre-generated dataset -----
def load_dataset(batch=0):
    global boards, player_hands, opponent_hands
    global player_flop_clusters, player_turn_clusters, player_river_clusters
    global opp_flop_clusters, opp_turn_clusters, opp_river_clusters
    global winners

    # Load the pre-generated dataset
    boards = np.load(f"dataset/boards_{batch}.npy").tolist()
    player_hands = np.load(f"dataset/player_hands_{batch}.npy").tolist()
    opponent_hands = np.load(f"dataset/opponent_hands_{batch}.npy").tolist()

    # Load player clusters
    player_flop_clusters = np.load(f"dataset/player_flop_clusters_{batch}.npy").tolist()
    player_turn_clusters = np.load(f"dataset/player_turn_clusters_{batch}.npy").tolist()
    player_river_clusters = np.load(f"dataset/player_river_clusters_{batch}.npy").tolist()

    # Load opponent clusters
    opp_flop_clusters = np.load(f"dataset/opp_flop_clusters_{batch}.npy").tolist()
    opp_turn_clusters = np.load(f"dataset/opp_turn_clusters_{batch}.npy").tolist()
    opp_river_clusters = np.load(f"dataset/opp_river_clusters_{batch}.npy").tolist()

    winners = np.load(f"dataset/winners_{batch}.npy")

    if max(player_flop_clusters) != NUM_FLOP_CLUSTERS - 1:
        raise ValueError(
            f"Expected {NUM_FLOP_CLUSTERS} clusters for player flop clusters, got {max(player_flop_clusters) + 1}"
        )
    if max(player_turn_clusters) != NUM_TURN_CLUSTERS - 1:
        raise ValueError(
            f"Expected {NUM_TURN_CLUSTERS} clusters for player turn clusters, got {max(player_turn_clusters) + 1}"
        )
    if max(player_river_clusters) != NUM_RIVER_CLUSTERS - 1:
        raise ValueError(
            f"Expected {NUM_RIVER_CLUSTERS} clusters for player river clusters, got {max(player_river_clusters) + 1}"
        )
    if max(opp_flop_clusters) != NUM_FLOP_CLUSTERS - 1:
        raise ValueError(
            f"Expected {NUM_FLOP_CLUSTERS} clusters for opponent flop clusters, got {max(opp_flop_clusters) + 1}"
        )
    if max(opp_turn_clusters) != NUM_TURN_CLUSTERS - 1:
        raise ValueError(
            f"Expected {NUM_TURN_CLUSTERS} clusters for opponent turn clusters, got {max(opp_turn_clusters) + 1}"
        )
    if max(opp_river_clusters) != NUM_RIVER_CLUSTERS - 1:
        raise ValueError(
            f"Expected {NUM_RIVER_CLUSTERS} clusters for opponent river clusters, got {max(opp_river_clusters) + 1}"
        )


# ----- Generate a dataset with associated clusters -----
def generate_dataset(num_samples=50000, batch=0, save=True):
    """
    To make things faster, we pre-generate the boards and hands. We also pre-cluster the hands
    """
    global boards, player_hands, opponent_hands
    global player_flop_clusters, player_turn_clusters, player_river_clusters
    global opp_flop_clusters, opp_turn_clusters, opp_river_clusters
    global winners

    boards, player_hands, opponent_hands = phEvaluatorSetup(num_samples)

    np_boards = np.array(boards)
    np_player_hands = np.array(player_hands)
    np_opponent_hands = np.array(opponent_hands)

    player_flop_cards = np.concatenate((np_player_hands, np_boards[:, :3]), axis=1).tolist()
    player_turn_cards = np.concatenate((np_player_hands, np_boards[:, :4]), axis=1).tolist()
    player_river_cards = np.concatenate((np_player_hands, np_boards), axis=1).tolist()
    opp_flop_cards = np.concatenate((np_opponent_hands, np_boards[:, :3]), axis=1).tolist()
    opp_turn_cards = np.concatenate((np_opponent_hands, np_boards[:, :4]), axis=1).tolist()
    opp_river_cards = np.concatenate((np_opponent_hands, np_boards), axis=1).tolist()

    print("generating clusters")

    player_flop_clusters = Parallel(n_jobs=-1)(
        delayed(predict_cluster)(cards) for cards in tqdm(player_flop_cards)
    )
    player_turn_clusters = Parallel(n_jobs=-1)(
        delayed(predict_cluster)(cards) for cards in tqdm(player_turn_cards)
    )
    player_river_clusters = Parallel(n_jobs=-1)(
        delayed(predict_cluster)(cards) for cards in tqdm(player_river_cards)
    )

    opp_flop_clusters = Parallel(n_jobs=-1)(
        delayed(predict_cluster)(cards) for cards in tqdm(opp_flop_cards)
    )
    opp_turn_clusters = Parallel(n_jobs=-1)(
        delayed(predict_cluster)(cards) for cards in tqdm(opp_turn_cards)
    )
    opp_river_clusters = Parallel(n_jobs=-1)(
        delayed(predict_cluster)(cards) for cards in tqdm(opp_river_cards)
    )

    winners = Parallel(n_jobs=-1)(
        delayed(evaluate_winner)(board, player_hand, opponent_hand)
        for board, player_hand, opponent_hand in tqdm(zip(boards, player_hands, opponent_hands))
    )

    if save:
        print("saving datasets")
        np.save(f"dataset/boards_{batch}.npy", boards)
        np.save(f"dataset/player_hands_{batch}.npy", player_hands)
        np.save(f"dataset/opponent_hands_{batch}.npy", opponent_hands)
        np.save(f"dataset/winners_{batch}.npy", winners)
        print("continuing to save datasets")

        np.save(f"dataset/player_flop_clusters_{batch}.npy", player_flop_clusters)
        np.save(f"dataset/player_turn_clusters_{batch}.npy", player_turn_clusters)
        np.save(f"dataset/player_river_clusters_{batch}.npy", player_river_clusters)

        np.save(f"dataset/opp_flop_clusters_{batch}.npy", opp_flop_clusters)
        np.save(f"dataset/opp_turn_clusters_{batch}.npy", opp_turn_clusters)
        np.save(f"dataset/opp_river_clusters_{batch}.npy", opp_river_clusters)


# Preflop Abstraction with 169 buckets (lossless abstraction)
def get_preflop_cluster_id(two_cards_string):  # Lossless abstraction for pre-flop, 169 clusters
    # cards input ex: Ak2h or ['Ak', '2h']
    """
    For the Pre-flop, we can make a lossless abstraction with exactly 169 buckets. The idea here is that what specific suits
    our private cards are doesn't matter. The only thing that matters is whether both cards are suited or not.

    This is how the number 169 is calculated:
    - For cards that are not pocket pairs, we have (13 choose 2) = 13 * 12 / 2 = 78 buckets (since order doesn't matter)
    - These cards that are not pocket pairs can also be suited, so we must differentiate them. We have 78 * 2 = 156 buckets
    - Finally, for cards that are pocket pairs, we have 13 extra buckets (Pair of Aces, Pair of 2, ... Pair Kings). 156 + 13 = 169 buckets

    Note that a pair cannot be suited, so we don't need to multiply by 2.

    Cluster ids:
    1-13 -> pockets
    14-91 -> Unsuited cluster pairs that are not pockets
    92-169 -> Suited cluster pairs that are not pockets

    """
    if type(two_cards_string) == list:
        two_cards_string = "".join(two_cards_string)

    assert len(two_cards_string) == 4

    KEY = {
        "A": 1,
        "2": 2,
        "3": 3,
        "4": 4,
        "5": 5,
        "6": 6,  # Supports both "T" and "10" as 10
        "7": 7,
        "8": 8,
        "9": 9,
        "T": 10,
        "10": 10,
        "J": 11,
        "Q": 12,
        "K": 13,
    }

    cluster_id = 0

    def hash_(a, b):
        """
        A2/2A -> 1
        A3/3A -> 2
        A4/4A -> 3
        ...
        KQ/QK -> 78

        returns values ranging from 1 to 78
        """
        assert a != b
        assert len(a) == 1 and len(b) == 1
        first = min(KEY[a], KEY[b])
        second = max(KEY[a], KEY[b])
        ans = first * (first - 1) / 2 + (second - 1)
        return int(ans)

    if two_cards_string[0] == two_cards_string[2]:  # pockets
        cluster_id = KEY[two_cards_string[0]]
    elif two_cards_string[1] != two_cards_string[3]:  # unsuited that are not pockets
        cluster_id = 13 + hash_(two_cards_string[0], two_cards_string[2])
    else:  # suited that are not pockets
        cluster_id = 91 + hash_(two_cards_string[0], two_cards_string[2])

    assert cluster_id >= 1 and cluster_id <= 169

    return cluster_id


def calculate_equity(player_cards: List[str], community_cards=[], n=2000, timer=False):
    if timer:
        start_time = time.time()
    wins = 0
    deck = fast_evaluator.Deck(excluded_cards=player_cards + community_cards)

    for _ in range(n):
        random.shuffle(deck)
        opponent_cards = deck[:2]  # To avoid creating redundant copies
        player_score = evaluate_cards(
            *(player_cards + community_cards + deck[2 : 2 + (5 - len(community_cards))])
        )
        opponent_score = evaluate_cards(
            *(opponent_cards + community_cards + deck[2 : 2 + (5 - len(community_cards))])
        )
        if player_score < opponent_score:
            wins += 1
        elif player_score == opponent_score:
            wins += 1
            # wins += 0.5

    if timer:
        print("Time it takes to call function: {}s".format(time.time() - start_time))

    return wins / n


def calculate_equity_distribution(
    player_cards: List[str], community_cards=[], bins=5, n=200, timer=False, parallel=False
):
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
    equity_hist = [
        0 for _ in range(bins)
    ]  # histogram of equities, where equity[i] represents the probability of the i-th bin

    assert len(community_cards) != 1 and len(community_cards) != 2

    deck = fast_evaluator.Deck(excluded_cards=player_cards + community_cards)

    def sample_equity():
        random.shuffle(deck)
        if len(community_cards) == 0:
            score = calculate_equity(player_cards, community_cards + deck[:3], n=200)
        elif len(community_cards) < 5:
            score = calculate_equity(player_cards, community_cards + deck[:1], n=100)
        else:
            score = calculate_equity(player_cards, community_cards, n=100)

        # equity_hist[min(int(score * bins), bins-1)] += 1.0 # Score of the closest bucket is incremented by 1
        return min(int(score * bins), bins - 1)

    if parallel:  # Computing these equity distributions in parallel is much faster
        equity_bin_list = Parallel(n_jobs=-1)(delayed(sample_equity)() for _ in range(n))

    else:
        equity_bin_list = [sample_equity() for _ in range(n)]

    for bin_i in equity_bin_list:
        equity_hist[bin_i] += 1.0

    # Normalize the equity so that the probability mass function (p.m.f.) of the distribution sums to 1
    for i in range(bins):
        equity_hist[i] /= n

    if timer:
        print("Time to calculate equity distribution: ", time.time() - start_time)
    return equity_hist


def plot_equity_hist(equity_hist, player_cards=None, community_cards=None):
    """Plot the equity histogram."""
    plt.clf()  # Clear Canvas
    plt.hist(
        [i / len(equity_hist) for i in range(len(equity_hist))],
        [i / len(equity_hist) for i in range(len(equity_hist) + 1)],
        weights=equity_hist,
    )
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
    plt.show(block=False)  # to plot graphs consecutively quickly with non-blocking behavior
    plt.pause(0.2)


# Post-Flop Abstraction using Equity Distributions
def create_abstraction_folders():
    if not os.path.exists("../data"):
        for split in ["clusters", "raw"]:
            for stage in ["flop", "turn", "river"]:
                os.makedirs(f"../data/{split}/{stage}")


def generate_postflop_equity_distributions(
    n_samples, bins, stage=None, save=True, timer=True
):  # Lossful abstraction for flop, turn and river
    if timer:
        start_time = time.time()
    assert stage is None or stage == "flop" or stage == "turn" or stage == "river"
    equity_distributions = []
    hands = []

    if stage is None:
        generate_postflop_equity_distributions(n_samples, bins, "flop", save, timer)
        generate_postflop_equity_distributions(n_samples, bins, "turn", save, timer)
        generate_postflop_equity_distributions(n_samples, bins, "river", save, timer)
    elif stage == "flop":
        num_community_cards = 3
    elif stage == "turn":
        num_community_cards = 4
    elif stage == "river":
        num_community_cards = 5

    deck = fast_evaluator.Deck()
    for i in tqdm(range(n_samples)):
        random.shuffle(deck)

        player_cards = deck[:2]
        community_cards = deck[2 : 2 + num_community_cards]
        distribution = calculate_equity_distribution(player_cards, community_cards, bins)
        equity_distributions.append(distribution)
        hands.append(" ".join(player_cards + community_cards))

    assert len(equity_distributions) == len(hands)

    equity_distributions = np.array(equity_distributions)
    print(equity_distributions)
    if save:
        create_abstraction_folders()
        file_id = int(time.time())  # Use the time as the file_id
        # TODO: Change to joblib saving for consistency everywhere
        with open(f"../data/raw/{stage}/{file_id}_samples={n_samples}_bins={bins}.npy", "wb") as f:
            np.save(f, equity_distributions)
        joblib.dump(
            hands, f"../data/raw/{stage}/{file_id}_samples={n_samples}_bins={bins}"
        )  # Store the list of hands, so you can associate a particular distribution with a particular hand


def get_filenames(folder, extension=".npy"):
    filenames = []

    for path in glob.glob(os.path.join(folder, "*" + extension)):
        # Extract the filename
        filename = os.path.split(path)[-1]
        filenames.append(filename)

    return filenames


def predict_cluster_kmeans(kmeans_classifier, cards, n=200):
    """cards is a list of cards"""
    assert type(cards) == list
    equity_distribution = calculate_equity_distribution(cards[:2], cards[2:], n=n)
    print(
        "averaged historgram: ",
        0.1 * equity_distribution[0]
        + 0.3 * equity_distribution[1]
        + 0.5 * equity_distribution[2]
        + 0.7 * equity_distribution[3]
        + 0.9 * equity_distribution[4],
    )
    y = kmeans_classifier.predict([equity_distribution])
    assert len(y) == 1
    return y[0]


def predict_cluster(cards):
    assert type(cards) == list

    if USE_KMEANS:
        if len(cards) == 5:  # flop
            return predict_cluster_kmeans(kmeans_flop, cards)
        elif len(cards) == 6:  # turn
            return predict_cluster_kmeans(kmeans_turn, cards)
        elif len(cards) == 7:  # river
            return predict_cluster_fast(cards, total_clusters=NUM_RIVER_CLUSTERS)
        else:
            raise ValueError("Invalid number of cards: ", len(cards))
    else:
        if len(cards) == 5:  # flop
            return predict_cluster_fast(cards, total_clusters=NUM_FLOP_CLUSTERS)
        elif len(cards) == 6:  # turn
            return predict_cluster_fast(cards, total_clusters=NUM_TURN_CLUSTERS)
        elif len(cards) == 7:  # river
            return predict_cluster_fast(cards, total_clusters=NUM_RIVER_CLUSTERS)
        else:
            raise ValueError("Invalid number of cards: ", len(cards))


def predict_cluster_fast(cards, n=2000, total_clusters=10):
    assert type(cards) == list
    equity = calculate_equity(cards[:2], cards[2:], n=n)
    cluster = min(total_clusters - 1, int(equity * total_clusters))
    return cluster


def load_kmeans_classifiers():
    global kmeans_flop, kmeans_turn, kmeans_river
    raw_dataset_filenames = sorted(get_filenames(f"../data/clusters/flop"))
    filename = raw_dataset_filenames[-1]  # Take the most recently generated dataset

    centroids = joblib.load(f"../data/clusters/flop/{filename}")
    kmeans_flop = KMeans(NUM_FLOP_CLUSTERS)
    kmeans_flop.cluster_centers_ = centroids
    kmeans_flop._n_threads = -1

    raw_dataset_filenames = sorted(get_filenames(f"../data/clusters/turn"))
    filename = raw_dataset_filenames[-1]  # Take the most recently generated dataset
    centroids = joblib.load(f"../data/clusters/turn/{filename}")
    kmeans_turn = KMeans(NUM_TURN_CLUSTERS)
    kmeans_turn.cluster_centers_ = centroids
    kmeans_turn._n_threads = -1

    raw_dataset_filenames = sorted(get_filenames(f"../data/clusters/river"))
    filename = raw_dataset_filenames[-1]  # Take the most recently generated dataset
    centroids = joblib.load(f"../data/clusters/river/{filename}")
    kmeans_river = KMeans(NUM_RIVER_CLUSTERS)
    kmeans_river.cluster_centers_ = centroids
    kmeans_river._n_threads = -1

    assert(len(kmeans_flop.cluster_centers_) == NUM_FLOP_CLUSTERS)
    assert(len(kmeans_turn.cluster_centers_) == NUM_TURN_CLUSTERS)
    assert(len(kmeans_river.cluster_centers_) == NUM_RIVER_CLUSTERS)

    return kmeans_flop, kmeans_turn, kmeans_river


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Poker Hand Abstractions.")
    parser.add_argument(
        "-g",
        "--generate",
        action="store_true",
        dest="generate",
        default=False,
        help="Generate Abstractions.",
    )
    parser.add_argument(
        "--n_samples",
        default=10000,
        dest="n_samples",
        help="Number of samples to sample from to generate the abstraction.",
    )
    parser.add_argument(
        "--n_clusters", default=50, dest="n_clusters", help="Number of clusters to generate."
    )
    parser.add_argument(
        "-b", "--bins", default=5, dest="bins", help="The granularity of your generated data."
    )
    parser.add_argument(
        "-s",
        "--stage",
        default="turn",
        dest="stage",
        help="Select the stage of the game that you would like to abstract (flop, turn, river).",
    )
    # Hyperparamtesrs
    args = parser.parse_args()

    generate = args.generate  # Generate histogram distributions to cluster on
    clustering = True  # Cluster these histogram distributions

    stage = args.stage
    n_samples = int(args.n_samples)
    bins = args.bins

    # TODO: River doesn't really need equity distribution... just calculate equity
    if generate:
        generate_postflop_equity_distributions(n_samples, bins, stage)

    if clustering:
        raw_dataset_filenames = sorted(get_filenames(f"../data/raw/{stage}"))
        filename = raw_dataset_filenames[
            -1
        ]  # Take the most recently generated dataset to run our clustering on

        equity_distributions = np.load(f"../data/raw/{stage}/{filename}")  # TODO: Switch to joblib
        print(filename)
        if not os.path.exists(f"../data/clusters/{stage}/{filename}"):
            print(f"Generating the cluster for the {stage}")
            print(filename)
            if stage == "flop":
                kmeans = KMeans(NUM_FLOP_CLUSTERS)
            elif stage == "turn":
                kmeans = KMeans(NUM_TURN_CLUSTERS)
            elif stage == "river":
                kmeans = KMeans(NUM_RIVER_CLUSTERS)
            else:
                raise ValueError("Invalid stage: ", stage)

            kmeans.fit(equity_distributions)  # Perform Clustering
            centroids = kmeans.cluster_centers_
            joblib.dump(centroids, f"../data/clusters/{stage}/{filename}")
        else:  # Centroids have already been generated, just load them, which are tensors
            centroids = joblib.load(f"../data/clusters/{stage}/{filename}")
            # Load KMeans Model
            if stage == "flop":
                kmeans = KMeans(NUM_FLOP_CLUSTERS)
            elif stage == "turn":
                kmeans = KMeans(NUM_TURN_CLUSTERS)
            elif stage == "river":
                kmeans = KMeans(NUM_RIVER_CLUSTERS)
            else:
                raise ValueError("Invalid stage: ", stage)

            kmeans.cluster_centers_ = centroids
            kmeans._n_threads = -1

        centroids = joblib.load(f"../data/clusters/{stage}/{filename}")
        # Load KMeans Model

    predict = False

    # # Visualization of the hands
    # hands = joblib.load(f'data/raw/{stage}/{filename.split(".")[0]}')
    # for i in range(equity_distributions.shape[0]):
    # 	hand = hands[i]
    # 	hand = hand.split(' ')
    # 	player_cards = hand[0]
    # 	community_cards = hand[0]
    # 	plot_equity_hist(equity_distributions[i], player_cards, community_cards)

    # Visualize the clusstering
