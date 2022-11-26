import json
import pytest

from leduc.state import Leduc as State
from leduc.state import Player
from leduc.hand_eval import leduc_eval
from leduc.vanilla import learn
from leduc.util import expected_utility
from leduc.monte import learn as mc_learn
from leduc.best_response import exploitability
from leduc.card import Card


def test_valid_actions():
	state = State([i for i in range(6)], 2, None)

	state.take('C')
	new_state = state.take('C', deep=True)

	assert (state.valid_actions() == ['F', 'C', '2R'] and
		   new_state.valid_actions() == ['F', 'C', '4R']), f'Actions {state.valid_actions()}'


	state = State([i for i in range(6)], 2, None)

	state = state.take('2R', deep=True)

	assert state.players[0].bets == 3, f'Bet {state.players[0].bets}'
	assert state.valid_actions() == ['F', 'C', '2R'], f'Actions {state.valid_actions()}'

	state = state.take('2R', deep=True)

	assert state.players[1].bets == 5, f'Bet {state.players[0].bets}'
	assert state.round == 0 and state.valid_actions() == ['F', 'C'], f'Actions {state.valid_actions()}'

	state = state.take('C', deep=True)

	assert all([p.bets == 5 for p in state.players]), f'Bet {state.players}'
	assert state.round == 1, f'Round {state.round}'


def test_terminal():
	state = State([i for i in range(6)], 2, None)

	state = state.take('F', deep=True)

	assert state.terminal == True, f'State {state}'

	state = State([i for i in range(6)], 2, None)
	state = state.take('C', deep=True)
	state = state.take('C', deep=True)
	state = state.take('C', deep=True)
	state = state.take('C', deep=True)

	assert state.terminal == True, f'State {state}'

	with pytest.raises(ValueError):
		state.take('1000R', deep=True)


def test_turn():
	state = State([i for i in range(6)], 2, None)

	assert state.turn == 0, f'State {state}'

	state = state.take('C', deep=True)
	assert state.turn == 1, f'State {state}'

	state = state.take('C', deep=True)
	assert state.turn == 0, f'State {state}'

	state = State([i for i in range(6)], 2, None)

	state = state.take('2R', deep=True)
	assert state.turn == 1, f'State {state}'

	state = state.take('2R', deep=True)
	assert state.turn == 0, f'State {state}'

	state = State([i for i in range(6)], 2, None)

	state = state.take('C', deep=True)
	assert state.turn == 1, f'State {state}'

	state = state.take('2R', deep=True)
	assert state.turn == 0, f'State {state}'

	state = state.take('2R', deep=True)
	assert state.turn == 1, f'State {state}'

def test_pot():
	state = State([i for i in range(6)], 2, None)

	assert sum(state.players) == state.num_players, sum(state.players)

	state = state.take('2R', deep=True)

	assert sum(state.players) == 4, sum(state.players)
	assert state.players[0].bets == 3, state.players[0].bets

	state = state.take('C', deep=True)

	assert sum(state.players) == 6, sum(state.players)
	assert state.players[1].bets == 3, state.players[1].bets

def test_player_add():
	p1 = Player()
	p2 = Player()
	
	assert sum([p1, p2]) == 2, sum([p1, p2])

	assert sum([Player() for _ in range(10)]) == 10, sum([Player() for _ in range(10)])

	assert exploit < .001, f"Exploitability was : {exploit}"