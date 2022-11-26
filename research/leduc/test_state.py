import numpy as np
import pytest

from leduc.state import State
from leduc.state import Leduc
from leduc.card import Card
from leduc.hand_eval import kuhn_eval, leduc_eval


def test_turn():
    state = State([1, 2, 3], 2, None)

    state.take('C')
    assert state.turn == 1

    state = State([1, 2, 3], 2, None)

    state.take('F')
    with pytest.raises(ValueError):
        state.take('C')


def test_terminal():
    state = State([1, 2, 3], 2, None)

    assert state.terminal is False, state

    state.take('F')

    assert state.terminal is True, state

    state = State([1, 2, 3], 3, None)

    assert state.terminal is False, state

    state.take('F')

    assert state.terminal is False, state

    state.take('F')

    assert state.terminal is True, state


def test_terminal_after_raise():
    state = State([1, 2, 3], 2, None)

    state = state.take('C', deep=True)
    state = state.take('1R', deep=True)

    assert state.terminal is False, state

    state = state.take('C', deep=True)

    assert state.terminal is True, state


def test_valid_actions():
    state = State([1, 2, 3], 2, None)

    actions = state.valid_actions()
    assert actions == ['F', 'C', '1R'], actions

    state.take('C')
    actions = state.valid_actions()
    assert actions == ['F', 'C', '1R'], actions

    state = State([1, 2, 3], 2, None)

    state.take('1R')
    actions = state.valid_actions()
    assert actions == ['F', 'C'], actions


def test_kuhn_utility():
    cards = [Card(14, 1), Card(13, 1), Card(12, 1)]
    state = State(cards, 2, kuhn_eval)

    state.take('C')
    state.take('C')

    utility = state.utility()

    assert np.array_equal(utility, np.array([1, -1])), utility

    cards = [Card(14, 1), Card(13, 1), Card(12, 1)]
    state = State(cards, 2, kuhn_eval)

    state.take('1R')
    state.take('C')

    utility = state.utility()

    assert np.array_equal(utility, np.array([2, -2])), utility


def test_hand_eval():
    cards = [Card(14, 1), Card(13, 1), Card(12, 1), Card(14, 2), Card(13, 2), Card(12, 2)]

    score = leduc_eval(cards[0], [cards[3]])

    assert score == 224


def test_leduc_state():
    cards = [Card(14, 1), Card(13, 1), Card(12, 1), Card(14, 2), Card(13, 2), Card(12, 2)]

    state = Leduc(cards, 2, leduc_eval)

    assert state.turn == 0, state.turn
    
    state = state.take('C', deep=True)

    assert state.turn == 1 and state.round == 0, f'{state.turn, state.round}'

    state = state.take('2R', deep=True)
    assert state.turn == 0 and state.round == 0, f'{state.turn, state.round}'

    state = state.take('C', deep=True)
    assert state.turn == 0 and state.round == 1, f'{state.turn, state.round}'

    state = state.take('F', deep=True)
    assert state.terminal is True and np.array_equal(state.utility(), np.array([-3, 3])), f'{state.terminal, state.utility()}'

    
def test_leduc_showdown():
    cards = [Card(14, 1), Card(13, 1), Card(12, 1), Card(14, 2), Card(13, 2), Card(12, 2)]

    state = Leduc(cards, 2, leduc_eval)

    state = state.take('2R', deep=True)
    state = state.take('2R', deep=True)
    state = state.take('C', deep=True)

    assert state.round == 1 and state.turn == 0, f'{state.round, state.turn}'

    state = state.take('4R', deep=True)
    
    assert state.turn == 1, state.turn
    
    state = state.take('C', deep=True)

    assert state.terminal is True and np.array_equal(state.utility(), np.array([9, -9])), f'{state.utility(), state.cards}'
    