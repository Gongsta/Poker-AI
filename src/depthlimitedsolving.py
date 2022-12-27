# This is the key code that will enable you to train a superhuman Poker AI.


"""
This code is inspired from the ideas of the original depth-limited solving for imperfect information games by Noam Brown. 
However, it is a different implementation that is likely much simpler, as it is written by an undergraduate
university student who doesn't know what he is doing.

ChatGPT:
1. Define the game tree for the game being solved, including all possible actions and their corresponding payoffs.
2. Initialize a "value" array to store the expected value of each subgame.
3. Iterate through the subgames in the game tree, starting at the leaf nodes and working backwards towards the root node. For each subgame, calculate the expected value using the value of the child subgames and the payoffs associated with each action.
4. After all subgames have been evaluated, the value of the root node will represent the expected value of the entire game. This can be used as an initial approximation of the optimal strategy.
5. To improve the solution, individual subgames can be solved in more detail by considering a larger portion of the game tree. This can be done by recursively applying the subgame solving algorithm to selected subgames.



"""
# Let me try the idea of subgame solving rock paper scissors.

import numpy as np
strategy = [0.1,0.8,0,1]

# TODO: Use 2 approaches, self-generative approach and bias approach
opp_strategy = [0.5, 0.2, 0.3]

class Node():
	def __init__(self, is_terminal=False) -> None:
		self.is_terminal = is_terminal
		pass
	
	def is_terminal(self):
		return self.is_terminal

def solve(node, strategy, memo):
	"""
	args:
		node: The node to be solved
		strategy: An approximation of the nash equilibrium strategy
		memo: A dictionary to store the expected values and optimal strategies of previously solved nodes
	"""
	# If the current node is a leaf node, return the payoff
	if node.is_terminal():
		return node.payoff, strategy
	
	# Check if the expected value and optimal strategy for the current node
	# have already been computed and stored in the memoization dictionary
	if node in memo:
		return memo[node]
	
	# Initialize the expected value and opponent expected value lists
	ev = []
	opp_ev = []
	
	# Iterate over the child nodes
	for i, child in enumerate(node.children):
		# Calculate the probability of reaching the child node
		probability = child.probability
		
		# Recursively solve the child node to get the expected value and strategy
		child_ev, child_strategy = solve(child, strategy, memo)
		
		# Append the expected value and opponent expected value to the lists
		ev.append(child_ev * probability)
		opp_ev.append(opp_child_ev * probability)
	
	# Compute the optimal strategy for the current node using the expected values
	# and opponent expected values of the child nodes
	optimal_strategy = compute_optimal_strategy(ev, opp_ev)
	
	# Store the expected value and optimal strategy for the current node in the memoization dictionary
	memo[node] = (sum(ev), optimal_strategy)
	
	# Return the expected value and optimal strategy
	return memo[node]



def compute_optimal_strategy(ev, opp_ev, num_strategies=10):
	# Calculate the net expected value for each action
	net_ev = [e - o for e, o in zip(ev, opp_ev)]
	
	# Sort the actions by net expected value
	sorted_actions = sorted(zip(net_ev, range(len(ev))), reverse=True)
	
	# Choose the top `num_strategies` actions
	top_actions = sorted_actions[:num_strategies]
	
	# Set the probabilities of the top actions to 1/num_strategies
	# TODO: This is bad, they just set a uniform distribution over "ok" actions, where len(EV) = # of actions
	optimal_strategy = [1/num_strategies if i in top_actions else 0 for i in range(len(ev))]
	
	return optimal_strategy
	
if __name__== "__main__":
	"""
	I need to spend some time thinking.
	
	Let's say that we came up with an equilibrium strategy for RPS that is [0.8, 0.1, 0.1].
	We obviously know that the equilibrium strategy is [0.33, 0.33, 0.33].
	
	so in the solve function, we want to calculate our expected values playing against different kinds of opponents,
	in the hopes of improving our blueprint strategy.
	
	If we have a strategy profile, we can calculate the expected values. (using Monte-Carlo simulations)
	
	The idea that chatGPT proposes is that we have the opponent select the actions that generate the highest expected values.
	And then then rinse and repeat. So you have around 10 strategies that the opponent has to try and exploit you.
	
	Now, you want to come up with a strategy that is more robust to the opponent trying to exploit you.
	Because your equilibrium strategy comes from an abstract game.
	
	But I am confused, because CFR is already doing that through regret minimization, it finds actions that minimize regrets.
	
	So isn't the idea to just keep running cfr on a subgame?


	SO how do you modify it?
	
	
	MISC: Ahh, with nested solving, for example if your opponent bets a weird amount, like 105$, instead
	of rounding it, you can solve it in a subtree using depth-limited solving and nested solving.
	
	Let's say that you opponent can come up with a set of strategies that generate a higher set of EV.
	How do you adjust to be robust? Because by adjusting, you will also be vulnerable to other.
	
	So then, you are creating something that 
	
	"""
