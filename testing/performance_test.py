"""
Performance testing to compare the speed of my hand evaluators vs the built-in libraries.
"""

import time
import random
import treys

def treysSetup(n, m):
    deck = treys.Deck()
    boards = []
    hands = []

    for i in range(n):
        boards.append(deck.draw(m))
        hands.append(deck.draw(2))
        deck.shuffle()

    return boards, hands


n = 10000
cumtime = 0.0
evaluator = treys.Evaluator()
boards, hands = treysSetup(n, 5)
for i in range(len(boards)):
    start = time.time()
    evaluator.evaluate(boards[i], hands[i])
    cumtime += (time.time() - start)

avg = float(cumtime / n)
print("7 card evaluation:")
print("[*] Treys: Average time per evaluation: %f" % avg)
print("[*] Treys: Evaluations per second = %f" % (1.0 / avg))

###

cumtime = 0.0
boards, hands = treysSetup(n, 4)
for i in range(len(boards)):
    start = time.time()
    evaluator.evaluate(boards[i], hands[i])
    cumtime += (time.time() - start)

avg = float(cumtime / n)
print("6 card evaluation:")
print("[*] Treys: Average time per evaluation: %f" % avg)
print("[*] Treys: Evaluations per second = %f" % (1.0 / avg))

###

cumtime = 0.0
boards, hands = treysSetup(n, 3)
for i in range(len(boards)):
    start = time.time()
    evaluator.evaluate(boards[i], hands[i])
    cumtime += (time.time() - start)

avg = float(cumtime / n)
print("5 card evaluation:")
print("[*] Treys: Average time per evaluation: %f" % avg)
print("[*] Treys: Evaluations per second = %f" % (1.0 / avg))