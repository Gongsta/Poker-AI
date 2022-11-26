import numpy as np

from itertools import permutations
from tqdm import tqdm

def expected_utility(cards, num_cards, num_players,
                     node_map, action_map):
    if len(cards) > 4:
        from leduc.state import Leduc as State
        from leduc.hand_eval import leduc_eval as eval
    else:
        from leduc.state import State
        from leduc.hand_eval import kuhn_eval as eval
    cards = sorted(cards)
    all_combos = [list(t) for t in set(permutations(cards, num_cards))]

    expected_utility = np.zeros(num_players)
    for card in tqdm(all_combos, desc='calculating expected utility'):
        hand = State(card, num_players, eval)
        expected_utility += traverse_tree(hand, node_map, action_map)

    return expected_utility/len(all_combos)


def traverse_tree(hand, node_map, action_map):
    if hand.terminal:
        utility = hand.utility()
        return utility

    info_set = hand.info_set()
    node = node_map[hand.turn][info_set]

    strategy = node.avg_strategy()
    util = np.zeros(len(node_map))
    valid_actions = action_map[hand.turn][info_set]
    if 'actions' in valid_actions:
        valid_actions = valid_actions['actions']
    for action in valid_actions:
        new_hand = hand.take(action, deep=True)
        util += traverse_tree(new_hand, node_map, action_map) * strategy[action]

    return util

    
def bias(strategy, action_to_bias):
    new_strat = {k:(v if k != action_to_bias else v * 5) for k, v in strategy.items()}

    norm_sum = sum([val for val in new_strat.values()])

    if norm_sum > 0:
        new_strat = {key: new_strat[key]/norm_sum for key in new_strat}
    else:
        num_valid = len(new_strat)
        new_strat = {key: 1/num_valid for key in new_strat}

    return new_strat