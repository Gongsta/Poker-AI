import sys
import json
import numpy as np

sys.path.append("../")

from itertools import permutations
from tqdm import tqdm
from leduc.best_response import exploitability
from leduc.node import Node
from leduc.card import Card
from leduc.util import expected_utility



def learn(iterations, cards, num_cards, node_map, action_map):
    if len(cards) > 4:
        from leduc.state import Leduc as State
        from leduc.hand_eval import leduc_eval as eval
    else:
        from leduc.state import State
        from leduc.hand_eval import kuhn_eval as eval
    all_combos = [list(t) for t in set(permutations(cards, num_cards))]
    num_players = len(node_map)
    for _ in tqdm(range(iterations), desc="learning"):
        card = np.random.choice(len(all_combos))
        state = State(all_combos[card], num_players, eval)
        probs = np.ones(num_players)
        accumulate_regrets(state, node_map, action_map, probs)


def accumulate_regrets(state, node_map, action_map, probs):
    if state.terminal:
        util = state.utility()
        return util

    info_set = state.info_set()

    if info_set not in action_map[state.turn]:
        action_map[state.turn][info_set] = state.valid_actions()

    valid_actions = action_map[state.turn][info_set]

    if info_set not in node_map[state.turn]:
        node_map[state.turn][info_set] = Node(valid_actions)

    node = node_map[state.turn][info_set]

    strategy = node.strategy(probs[state.turn])

    util = {a: 0 for a in valid_actions}
    node_util = np.zeros(len(node_map))
    for action in valid_actions:
        new_prob = [p if i != state.turn else p*strategy[action]
                    for i, p in enumerate(probs)]
        new_state = state.take(action, deep=True)
        returned = accumulate_regrets(new_state, node_map,
                                      action_map, new_prob)

        util[action] = returned[state.turn]
        node_util += returned * strategy[action]

    reach_prob = 1
    for p, prob in enumerate(probs):
        if p != state.turn:
            reach_prob *= prob

    for action in valid_actions:
        regret = util[action] - node_util[state.turn]
        node.regret_sum[action] += regret * reach_prob

    return node_util


if __name__ == '__main__':

    num_players = 2
    node_map = {i: {} for i in range(num_players)}
    action_map = {i: {} for i in range(num_players)}
    cards = [Card(14, 1), Card(13, 1), Card(12, 1), Card(14, 2), Card(13, 2), Card(12, 2)]
    learn(10000, cards, 3, node_map, action_map)

    for player in node_map:
        print(f"Player {player}")
        print('Number of info sets', len(node_map[player]))
        for info_set, node in node_map[player].items():
            avg_strat = node.avg_strategy()
            print(f"{info_set}: {avg_strat}")
        

    util = expected_utility(cards, 3, 2, node_map, action_map)
    print(util)