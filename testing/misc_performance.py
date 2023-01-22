# I wanted to know the speed difference between OOP and procedural
class Node:
	def __init__(self, history: str):
		self.history = history
	
	def is_terminal(self):
		return self.history[-1] ==  "f"


def is_terminal(history: str):
	return history[-1] == "f"
	
import time
import numpy as np

def test_oop_vs_procedural():
	alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
	start = time.time()
	for i in range(100000):
		character = np.random.choice(alphabet)
		is_terminal("f")

	print(f"time for procedural: {time.time()- start}s")

	start = time.time()
	for i in range(100000):
		character = np.random.choice(alphabet)
		node = Node(character)
		node.is_terminal()

	print(f"time for OOP: {time.time()- start}s")
	
	"""
	Results:
	time for procedural: 0.6510379314422607s
	time for OOP: 0.6217091083526611s
	
	I am so confused, why is OOP faster than procedural?? It would make sense that procedural is faster. 
	"""

def test_binary_vs_modulus():
	start = time.time()
	x = 0
	for i in range(100_000):
		x = not x

	print(f"Time for binOp: {time.time() - start}s")

	for i in range(100_000):
		x = (x + 1) % 2
	print(f"Time for modulus: {time.time() - start}s")

	"""
	Results:
	Time for binOp: 0.0013861656188964844s
	Time for modulus: 0.011712074279785156s

	"""

import sys
sys.path.append('../src')

from abstraction import *

def test_inference():
	start = time.time()
	kmeans_flop, kmeans_turn, kmeans_river = load_kmeans_classifiers()
	print(f"Time to load kmeans: {time.time() - start}s")

	start = time.time()
	for i in range(1):
		get_flop_cluster_id(kmeans_flop, "AhAd2s2d3h")
	print(f"Average Time to predict flop cluster id: {(time.time() - start)/1}s")


	start = time.time()
	for i in range(10):
		get_turn_cluster_id(kmeans_turn, "AhAd2s2d3h4s")
	print(f"Average Time to predict turn cluster id: {(time.time() - start)/100}s")

	start = time.time()
	for i in range(10):
		get_river_cluster_id(kmeans_river, "AhAd2s2d3h4s5s")
	print(f"Average Time to predict river cluster id: {(time.time() - start)/1000}s")
	
	"""
	Results (Prior to optimization)
	Time to load kmeans: 0.0005240440368652344s
	Average Time to predict flop cluster id: 2.9904518127441406s
	Average Time to predict turn cluster id: 0.006986420154571533s
	Average Time to predict river cluster id: 0.0006327447891235352s
	
	
	"""


if __name__ == "__main__":
	# test_oop_vs_procedural()
	# test_binary_vs_modulus()
	test_inference()