"""
Performance testing to compare the speed of my hand evaluators vs the built-in libraries.
"""

import time
import random
import treys
import sys
from phevaluator import evaluate_cards

sys.path.append("../src")
from environment import *
from evaluator import *


def treysSetup(n):
    deck = treys.Deck()

    boards = []
    player_hands = []
    opponent_hands = []
    for _ in range(n):
        boards.append(deck.draw(5))
        player_hands.append(deck.draw(2))
        opponent_hands.append(deck.draw(2))
        deck.shuffle()

    return boards, player_hands, opponent_hands


def phEvaluatorSetup(n):
    deck = []

    def shuffle_deck():
        deck.clear()
        for rank in ["A", "2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K"]:
            for suit in ["h", "d", "s", "c"]:
                deck.append(rank + suit)
        random.shuffle(deck)

    boards = []
    player_hands = []
    opponent_hands = []
    for _ in range(n):
        shuffle_deck()
        boards.append(deck[:5])
        player_hands.append(deck[5:7])
        opponent_hands.append(deck[7:9])

    return boards, player_hands, opponent_hands


def customSetup(n):  # Test the speed of our own evaluator (it's probably terrible)
    deck = Deck()

    boards = []
    player_hands = []
    opponent_hands = []
    for _ in range(n):
        player_hands.append([deck.draw(), deck.draw()])
        opponent_hands.append([deck.draw(), deck.draw()])
        boards.append([deck.draw(), deck.draw(), deck.draw(), deck.draw(), deck.draw()])
        deck.reset_deck()

    return boards, player_hands, opponent_hands


n = 10000
cumtime = 0.0
evaluator = treys.Evaluator()
start = time.time()
boards, player_hands, opponent_hands = treysSetup(n)
for i in range(len(boards)):
    # start = time.time()
    wins = 0
    losses = 0
    ties = 0
    p1_score = evaluator.evaluate(player_hands[i], boards[i])
    p2_score = evaluator.evaluate(opponent_hands[i], boards[i])
    if p1_score > p2_score:
        wins += 1
    elif p1_score < p2_score:
        losses += 1
    else:
        ties += 1

cumtime += time.time() - start

avg = float(cumtime / n)
print("7 card evaluation:")
print("[*] Treys: Average time per evaluation: %f" % avg)
print("[*] Treys: Evaluations per second = %f" % (1.0 / avg))

n = 10000
cumtime = 0.0
start = time.time()
boards, player_hands, opponent_hands = phEvaluatorSetup(n)
wins = 0
losses = 0
ties = 0
for i in range(len(boards)):

    p1_score = evaluate_cards(*(boards[i] + player_hands[i]))
    p2_score = evaluate_cards(*(boards[i] + opponent_hands[i]))
    if p1_score > p2_score:
        wins += 1
    elif p1_score < p2_score:
        losses += 1
    else:
        ties += 1
cumtime += time.time() - start

avg = float(cumtime / n)
print("7 card evaluation:")
print("[*] PhEvaluator: Average time per evaluation: %f" % avg)
print("[*] PhEvaluator: Evaluations per second = %f" % (1.0 / avg))


n = 100
evaluator = Evaluator()
start = time.time()
boards, player_hands, opponent_hands = customSetup(n)
for i in range(len(boards)):
    wins = 0
    losses = 0
    ties = 0
    evaluator.add_hands(CombinedHand(boards[i] + player_hands[i]))
    evaluator.add_hands(CombinedHand(boards[i] + opponent_hands[i]))
    winner = evaluator.get_winner()

cumtime += time.time() - start

avg = float(cumtime / n)
avg = float(cumtime / n)
print("7 card evaluation:")
print("[*] Custom Evaluator: Average time per evaluation: %f" % avg)
print("[*] Custom Evaluator: Evaluations per second = %f" % (1.0 / avg))
