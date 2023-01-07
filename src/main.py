"""
Main code that runs all parts of the game.

Two modes:
1. Training: trains the AI to play the game
2. Playing: play the game against the AI

Training Pipeline:
1. Generate abstractions for the holdem game
2. Use Monte-Carlo CFR to generate a blueprint strategy for the abstracted game

Playing Pipeline:
1. Use the blueprint strategy + real-time search (depth-limited solving) to play the game
2. Integrate with pygame
"""


if __name__ == "__main__":
	