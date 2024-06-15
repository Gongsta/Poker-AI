# Regret Matching Algorithm Implemented from this paper: http://modelai.gettysburg.edu/2013/cfr/cfr.pdf
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
import sys

sys.path.append("../../")
from helper import animate_rps

ROCK, PAPER, SCISSORS, NUM_ACTIONS = 0, 1, 2, 3
regretSum = np.zeros(NUM_ACTIONS)
strategy = np.zeros(NUM_ACTIONS)
strategySum = np.zeros(NUM_ACTIONS)
oppStrategy = np.array([0.4, 0.3, 0.3])
strategy_hist = []


def getStrategy():
    normalizingSum = 0
    for a in range(NUM_ACTIONS):
        strategy[a] = max(regretSum[a], 0)
        normalizingSum += strategy[a]

    for a in range(NUM_ACTIONS):
        if normalizingSum > 0:
            strategy[a] /= normalizingSum
        else:
            strategy[a] = 1 / NUM_ACTIONS

        strategySum[a] += strategy[a]

    return strategy


def getAction(strategy):
    r = np.random.random()
    cumulativeProbability = 0
    action = 0
    for a in range(NUM_ACTIONS):
        action = a
        cumulativeProbability += strategy[a]
        if r < cumulativeProbability:
            break

    return action


def train(iterations: int, visualize=False):
    actionUtility = np.zeros(NUM_ACTIONS)

    for i in range(iterations):
        # Get regret-matched mixed-strategy actions
        strategy = getStrategy()

        if visualize:
            avgStrategy = getAverageStrategy()
            strategy_hist.append(
                deepcopy(avgStrategy)
            )  # Append to the history of strategies so we can visualize it
        myAction = getAction(strategy)
        otherAction = getAction(oppStrategy)

        # Compute action utilities
        actionUtility[otherAction] = 0
        actionUtility[(otherAction + 1) % NUM_ACTIONS] = 1
        actionUtility[(otherAction - 1) % NUM_ACTIONS] = -1

        # Accumulate action regrets
        for a in range(NUM_ACTIONS):
            regretSum[a] += actionUtility[a] - actionUtility[myAction]

    if visualize:
        animate_rps(strategy_hist)


def getAverageStrategy():
    avgStrategy = np.zeros(NUM_ACTIONS)
    normalizingSum = 0
    for a in range(NUM_ACTIONS):
        normalizingSum += strategySum[a]

    for a in range(NUM_ACTIONS):
        if normalizingSum > 0:
            avgStrategy[a] = strategySum[a] / normalizingSum
        else:
            avgStrategy[a] = 1 / NUM_ACTIONS

    return avgStrategy


def main():
    train(100000, visualize=True)
    avgStrategy = getAverageStrategy()
    print("Strategy computed: ", avgStrategy)
    print("Sum of the probabilities should be 1: ", avgStrategy.sum())


if __name__ == "__main__":
    main()
