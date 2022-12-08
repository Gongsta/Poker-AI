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




if __name__ == "__main__":
	test_binary_vs_modulus()