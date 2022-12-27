import base
from base import Player, Action
from typing import NewType, Dict, List, Callable, cast
import copy

class RPSHistory(base.History):
	def __init__(self, history: List[Action] = []):
		self.history = history
	
	def is_terminal(self):
		return len(self.history) == 2
	
	def actions(self):
		return ['R', 'P', 'S']
	
	def player(self):
		plays = len(self.history)
		return plays % 2
	
	def terminal_utility(self, i: Player) -> int:
		assert(self.is_terminal())
		p1_choice = self.history[0]
		p2_choice = self.history[1]
		
		p1_idx = "RPS".index(p1_choice)
		p2_idx = "RPS".index(p2_choice)
		
		if p1_idx == p2_idx:
			return 0
		elif (p1_idx + 1) % 3 == p2_idx:
			return -1 if i == 0 else 1
		else:
			return 1 if i == 0 else -1
		
	def __add__(self, action: Action):
		return RPSHistory(self.history + [action])
	
	def get_infoSet_key(self) -> List[Action]:
		history = copy.deepcopy(self.history)
		if len(history) >= 1:
			history[0] = '?'
		return history

class RPSInfoSet(base.InfoSet):
	def __init__(self, infoSet: List[Action]):
		super().__init__(infoSet)

	def actions(self) -> List[Action]:
		return ['R', 'P', 'S']
	
	def player(self) -> Player:
		plays = len(self.infoSet)
		return plays % 2
	
def create_infoSet(infoSet_key: List[Action]):
	"""
	We create an information set from a history. 
	"""
	return RPSInfoSet(infoSet_key)
	
	
def create_history():
	return RPSHistory()
	

if __name__ == "__main__":
	cfr = base.CFR(create_infoSet, create_history, iterations=1_00_000)
	# cfr.solve()
	# print(cfr.get_expected_value(RPSHistory([]), 0, player_strategy=[1,0,0], opp_strategy=[0.5,0.5,0]))
	print(cfr.get_best_response(RPSHistory([]), 0, player_strategy=[0.8,0.2,0]))