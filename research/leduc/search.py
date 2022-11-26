import glob
import pickle
import numpy as np
from copy import deepcopy
import sys

sys.path.append('../')

from leduc.card import Card
from leduc.node import MNode as Node
from leduc.monte import learn, Search
from itertools import permutations
from leduc.state import Leduc as State
from leduc.hand_eval import leduc_eval as eval


class Pluribus:
    def __init__(self, node_map, action_map, cards, num_cards):
        self.blueprint = node_map
        self.action_map = action_map

        self.all_combos = [list(t) for t in set(permutations(cards, num_cards))]
        card = np.random.choice(len(self.all_combos))
        self.root = State(self.all_combos[card], len(node_map), eval) 


    def play(self):
        self.node_map = deepcopy(self.blueprint)
        actions = self.action_map


        pluribus = 0
        state = deepcopy(self.root)
        while state.terminal is False:
            player_turn = state.turn

            if player_turn == pluribus:
                self.pluribus_turn(state, self.node_map, actions, cards)

            else:
                while True:
                    action = input("Choose an Action: F, C, $R, where $ is any integer: ")
                    if action in ['F', 'C'] or (action[0].isnumeric() and len(action) > 1 and action[1] == 'R'):
                        break
                    else:
                        print("Please choose a valid action (F, C, $R)")

                self.opponent_turn(action, state, self.blueprint, actions, cards)

        payout = state.utility()
        print(f"Game state {state}")
        print(state.cards)
        if payout[0] > 0:
            print(f"Pluribus won {payout[0]} chips")
        elif payout[0] < 0:
            print(f"You won {payout[1]} chips")
        else:
            print(f"There was a tie!")
            
                 
    def pluribus_turn(self, state, blueprint, action_map, cards):
        info_set = state.info_set()
        turn = state.turn
        if info_set not in action_map[turn]:
            action_map[turn][info_set] = {'actions': state.valid_actions()}

        valid_actions = action_map[turn][info_set]['actions']
        if state.info_set() not in blueprint[state.turn]:
            blueprint[state.turn][state.info_set()] = Node(valid_actions)

        node = blueprint[state.turn][state.info_set()]
        strategy = node.avg_strategy()

        actions = list(strategy.keys())
        probs = list(strategy.values())

        sampled = actions[np.random.choice(len(actions), p=probs)]
        print(f"Pluribus played {sampled}")

        action_map[state.turn][state.info_set()]['frozen'] = sampled

        state.take(sampled)

        self.check_round(state, self.root, blueprint, action_map, cards)    


    def opponent_turn(self, action, state, blueprint, actions, cards):
        node_map = None
        state.take(action)
        info_set = state.info_set()
        turn = state.turn
        if info_set not in actions[turn]:
            actions[turn][info_set] = {'actions': state.valid_actions()}

        if action not in actions[turn][info_set]['actions']:
            for info_set in actions[state.turn]:
                if str(state) in info_set:
                    actions[state.turn][info_set]['actions'].append(action) 

            search = Search(self.root, blueprint, actions, cards, len(state.cards))
            print("***Action not found, finding strategy to counter***")
            self.node_map = search.search()

        self.check_round(state, self.root, node_map if node_map is not None else blueprint,
                    actions, cards)
                    

    def check_round(self, next_state, state, blueprint, actions, cards):
        if next_state.round > state.round:
            self.root = next_state
            search = Search(next_state, blueprint, actions, cards, len(state.cards))
            print("***Reached end of round, updating strategy***")
            new_strat = search.search()
            self.node_map = new_strat


if __name__ == "__main__":
    if not glob.glob('blueprint.po'):
        num_players = 2
        node_map = {i: {} for i in range(num_players)}
        action_map = {i: {} for i in range(num_players)}
        cards = [Card(14, 1), Card(13, 1), Card(12, 1), Card(14, 2), Card(13, 2), Card(12, 2)]
        learn(50000, cards, 3, node_map, action_map)
        with open('blueprint.po', 'wb') as f:
            pickle.dump(node_map, f)
        with open('actions.po', 'wb') as f:
            pickle.dump(action_map, f)

    else: # Load Blueprint Strategy
        with open('blueprint.po', 'rb') as f:
            node_map = pickle.load(f)

        with open('actions.po', 'rb') as f:
            action_map = pickle.load(f)

    cards = [Card(14, 1), Card(13, 1), Card(12, 1), Card(14, 2), Card(13, 2), Card(12, 2)]
    pluribus = Pluribus(node_map, action_map, cards, 3)
    pluribus.play()