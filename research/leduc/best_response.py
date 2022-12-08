import numpy as np

from itertools import permutations

def exploitability(cards, num_cards, node_map, action_map):
    if len(cards) > 4:
        from leduc.state import Leduc as State
        from leduc.hand_eval import leduc_eval as eval
    else:
        from leduc.state import State
        from leduc.hand_eval import kuhn_eval as eval

    public_states, start = build_tree(cards, len(node_map))
    exploit = 0 
    for player in range(len(node_map)):
        v = expectimax(start, public_states, cards, player, node_map, 1)
        exploit += v

    return exploit/len(node_map)


def build_tree(cards, num_players):
    if len(cards) > 4:
        from leduc.state import Leduc as State
        from leduc.hand_eval import leduc_eval as eval
    else:
        from leduc.state import State
        from leduc.hand_eval import kuhn_eval as eval
    
    state = State(cards, num_players, eval)
    public_states = {} 

    traverse_public(state, public_states)

    return public_states, state


def traverse_public(state, public):
    if state.terminal:
        return

    for action in state.valid_actions():
        new_state = state.take(action, deep=True)
        if state not in public:
            public[state] = {}
        public[state][action] = new_state
        traverse_public(new_state, public)
    
    return

    
def expectimax(public_state, state_map, cards, fixed, node_map, prob):
    if public_state.terminal:
        # normalize prob for everyone else
        all_deals = [list(t) for t in set(permutations(cards, 2))]
        util = np.zeros(len(node_map)) 
        for deal in all_deals:
            public_state.cards = deal
            util += public_state.utility() * prob
        return util[fixed]

    v = float('-inf')
    valid_actions = public_state.valid_actions()
    v_util = {a: 0 for a in valid_actions}
    w = {a: 0 for a in valid_actions}
    new_prob = prob
    for action, state in state_map[public_state].items():
        if public_state.turn != fixed:
            # compute weight
            new_prob = compute_weight(public_state, action, node_map, prob)
            w[action] = new_prob
        v_util[action] = expectimax(state, state_map, cards, fixed, node_map, new_prob)
        if public_state.turn == fixed and v_util[action] > v:
            v = v_util[action]
            
    if public_state.turn != fixed:
       norm = normalize(w) 
       if v == float('-inf'):
           v = 0
    
       for action in state_map[public_state]:
           v += norm[action] * v_util[action]
           
    return v
        

def normalize(map):
    norm_sum = sum(map.values())

    if norm_sum > 0:
        normed = {key: map[key]/norm_sum for key in map}
    else:
        num_valid = len(map)
        normed = {key: 1/num_valid for key in map}
        
    return normed


def compute_weight(state, action, node_map, prob):
    player = state.turn
    nodes = node_map[player]
    next_state = state.take(action, deep=True)

    for info_set in nodes:
        if str(state.history) in info_set:
            prob *= nodes[info_set].avg_strategy()[action]
            
    return prob