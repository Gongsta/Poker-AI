import base
from base import Player, Action
import random
from typing import NewType, Dict, List, Callable, cast
import copy


class KuhnHistory(base.History):
    """
    Example of history:
    First two actions are the cards dealt to the players. The rest of the actions are the actions taken by the players.
            1. ['1', '2', 'b', 'p', 'b', 'p']
            2. ['2', '3', 'p', 'p']

    """

    def __init__(self, history: List[Action] = []):
        super().__init__(history)

    def is_terminal(self):
        plays = len(self.history)

        if plays > 3:
            player_card = self.history[0]
            opponent_card = self.history[1]

            assert player_card in ["1", "2", "3"]
            assert opponent_card in ["1", "2", "3"]
            assert player_card != opponent_card

            terminalPass = self.history[-1] == "p"
            doubleBet = self.history[-2:] == ["b", "b"]

            if terminalPass or doubleBet:
                return True

            else:
                return False

    def actions(self):
        if self.is_chance():
            if len(self.history) == 0:
                return ["1", "2", "3"]
            else:
                cards = ["1", "2", "3"]
                cards.remove(self.history[0])  # Two players cannot get the same cards
                return cards

        elif not self.is_terminal():
            return ["p", "b"]

        else:
            raise Exception("No actions available for terminal history")

    def player(self):
        plays = len(self.history)
        if plays <= 1:
            return -1
        else:
            return plays % 2

    def is_chance(self):
        return super().is_chance()

    def sample_chance_outcome(self):
        assert self.is_chance()

        cards = self.actions()
        return random.choice(cards)  # Sample one of the cards with equal probability

    def terminal_utility(self, i: Player) -> int:
        assert self.is_terminal()  # We can only call the utility for a terminal history
        assert i in [0, 1]  # Only works for 2 player games for now

        terminalPass = self.history[-1] == "p"
        doubleBet = self.history[-2:] == ["b", "b"]

        player_card = self.history[i % 2]
        opponent_card = self.history[(i + 1) % 2]
        is_player_winner = player_card > opponent_card

        if terminalPass:
            if self.history[-2:] == ["p", "p"]:
                return 1 if is_player_winner else -1
            else:
                if len(self.history) % 2 == 0:  # i.e.: bp
                    return 1 if i == 0 else -1
                else:  # i.e.: pbp
                    return 1 if i == 1 else -1
        else:
            return 2 if is_player_winner else -2

    def __add__(self, action: Action):
        new_history = KuhnHistory(self.history + [action])
        return new_history

    def get_infoSet_key(self) -> List[Action]:
        assert not self.is_chance()  # chance history should not be infosets
        assert not self.is_terminal()  # terminal history is not an infoset

        player = self.player()
        if player == 0:
            history = copy.deepcopy(self.history)
            history[1] = "?"  # Unknown card
            return history
        else:
            history = copy.deepcopy(self.history)
            history[0] = "?"  # Unknown card
            return history


class KuhnInfoSet(base.InfoSet):
    """
    Information Sets (InfoSets) cannot be chance histories, nor terminal histories.
    This condition is checked when infosets are created.

    """

    def __init__(self, infoSet_key: List[Action], actions: List[Action], player: Player):
        assert len(infoSet_key) >= 2
        super().__init__(infoSet_key, actions, player)


def create_infoSet(infoSet_key: List[Action], actions: List[Action], player: Player):
    """
    We create an information set from a history.
    """
    return KuhnInfoSet(infoSet_key, actions, player)


def create_history(t):
    return KuhnHistory()


if __name__ == "__main__":
    cfr = base.CFR(create_infoSet, create_history, iterations=5000)
    cfr.solve(debug=False, method="vanilla")
    # TODO: Add playing option, right now there is old code in research/kuhn,
    # which is not oop
