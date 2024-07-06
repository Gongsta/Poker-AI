import base
from base import Player, Action
from typing import List
import random


class CoinTossHistory(base.History):
    """
    ['H', 'P', 'H']
    """

    def __init__(self, history: List[Action] = []):
        super().__init__(history)

    def is_terminal(self):
        if len(self.history) == 3 or (len(self.history) == 2 and self.history[-1] == "Fold"):
            return True
        else:
            return False

    def actions(self):
        if self.is_chance():  # draw cards
            return ["Heads", "Tails"]

        elif not self.is_terminal():
            if len(self.history) == 1:
                return ["Fold", "Play"]
            else:
                return ["Fold", "Heads", "Tails"]

        else:
            raise Exception("Cannot call actions on a terminal history")

    def player(self):
        if len(self.history) == 0:
            return -1
        else:
            return (len(self.history) + 1) % 2

    def is_chance(self):
        return super().is_chance()

    def sample_chance_outcome(self):
        assert self.is_chance()
        return random.choice(["Heads", "Tails"])

    def terminal_utility(self, i: Player) -> int:
        assert self.is_terminal()  # We can only call the utility for a terminal history
        assert i in [0, 1]  # Only works for 2 player games for now

        util = 0

        if len(self.history) == 2:
            if self.history[0] == "Heads":
                util = 0.5
            else:
                util = -0.5
        else:
            assert(len(self.history) == 3)
            if self.history[-1] == "Fold":
                util = 1
            elif self.history[0] == self.history[-1]:  # Correct guess by opponent
                util = -1
            else:  # incorrect guess by opponent
                util = 1

        if i == 0:
            return util
        else:
            return -util

    def __add__(self, action: Action):
        new_history = CoinTossHistory(self.history + [action])
        return new_history

    def get_infoSet_key(self) -> List[Action]:
        """
        This is where we abstract away cards and bet sizes.
        """
        assert not self.is_chance()
        assert not self.is_terminal()

        player = self.player()
        if player == 0:
            return self.history
        else:
            return self.history[1:]


class CoinTossInfoSet(base.InfoSet):
    def __init__(self, infoSet_key: List[Action], actions: List[Action], player: Player):
        assert len(infoSet_key) >= 1
        super().__init__(infoSet_key, actions, player)

def create_infoSet(infoSet_key: List[Action], actions: List[Action], player: Player):
    """
    We create an information set from a history.
    """
    return CoinTossInfoSet(infoSet_key, actions, player)


def create_history(sample_id):
    return CoinTossHistory()


class CoinTossCFR(base.CFR):
    def __init__(
        self,
        create_infoSet,
        create_history,
        n_players: int = 2,
        iterations: int = 1000000,
    ):
        super().__init__(create_infoSet, create_history, n_players, iterations)


if __name__ == "__main__":
    ITERATIONS = 100
    cfr = CoinTossCFR(create_infoSet, create_history, iterations=ITERATIONS)
    cfr.solve(debug=False, method="vanilla")
    cfr.export_infoSets("cointoss.joblib")
