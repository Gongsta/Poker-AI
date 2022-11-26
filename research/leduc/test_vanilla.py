import json
import numpy as np


from leduc.vanilla import learn
from leduc.util import expected_utility
from leduc.best_response import exploitability
from leduc.card import Card


def test_expected_utility():
    num_players = 2
    node_map = {i: {} for i in range(num_players)}
    action_map = {i: {} for i in range(num_players)}
    cards = [Card(14, 1), Card(13, 1), Card(12, 1)]
    learn(10000, cards, 2, node_map, action_map)
    print(node_map)
    util = expected_utility(cards, 2, 2, node_map, action_map)
    print(util)

    exploit = exploitability(cards, 2, node_map, action_map)
    print(exploit)

    assert exploit < .001 and exploit != float('-inf'), f"Exploitability was : {exploit}"

    print(json.dumps(action_map, indent=4))
    print(node_map)

    assert util.sum() == 0
    assert np.isclose(np.abs(util[0]), np.abs(util[1])), f"Util was {util}"
    assert np.abs(util).sum() > 0, f"Util was {util}"


def test_expected_utility_three_player():
    num_players = 3
    node_map = {i: {} for i in range(num_players)}
    action_map = {i: {} for i in range(num_players)}
    cards = [Card(14, 1), Card(13, 1), Card(12, 1), Card(11, 1)]
    learn(10000, cards, 3, node_map, action_map)

    util = expected_utility(cards, 3, 3, node_map, action_map)

    print(util)
    print(json.dumps(action_map, indent=4))
    print(node_map)

    assert abs(util.sum()) <= 0.0001, f"Something weird, not a zero sum game"
    assert np.abs(util).sum() > 0, f"Util was {util}"
