"""
Lossless abstraction of preflop. Since we only have a single stage, we can be more granular with the bet abstraction.

169 * 11 = 1859 possible sequences

When training, we assume 1 BB = 2 chips.

Preflop is weird because the dealer calls, but the round isn't over...

"""

import base
from base import Player, Action
import abstraction
from typing import List
from abstraction import (
    get_preflop_cluster_id,
)
from fast_evaluator import evaluate_cards

DISCRETE_ACTIONS = ["k", "bMIN", "bMID", "bMAX", "c", "f"]

player_hands = None
opponent_hands = None
boards = None
winners = None


class PreflopHoldemHistory(base.History):
    """
    Example of history:
    First two actions are the cards dealt to the players. Then, beting round. Then all community cards are shown, and winner is decided.
            # non-dealer, dealer, dealer,
            1. ['AkTh', 'QdKd', 'c', 'k', '/', 'Qh2d3s4h5s']

    Non-dealer, then dealer, because that is the order of play for the rest of the game. yes, it's confusing

    Infoset:
    [150, 'k', 'k']

    ---- ACTIONS ----
    - k = check
    - bX = bet X amount (this includes raising)
    - c = call
    - f = fold (you cannot fold if the other player just checked)

    """

    def __init__(self, history: List[Action] = [], sample_id=0):
        super().__init__(history)
        self.sample_id = sample_id

    def is_terminal(self):
        if len(self.history) == 0:
            return False
        if len(self.history[-1]) == 10:  # show community cards
            return True
        else:
            return False

    def actions(self):
        if self.is_chance():  # draw cards
            return (
                []
            )  # This should return the entire deck with current cards removed, but I do this for speedup by loading an existing dataset

        elif not self.is_terminal():
            """
            To limit this game going to infinity, I only allow these betting seqeunces:
            Else the branching factor huge.
            ck
            cbMINf
            cbMINc
            cbMIDf
            cbMIDc
            cbMAXf
            cbMAXc
            bMINf
            bMINc
            bMIDf
            bMIDc
            bMINbMAXf
            bMINbMAXc
            bMINbMIDf
            bMINbMIDc
            bMINbMIDbMAXc
            bMINbMIDbMAXf
            bMIDbMAXc
            bMIDbMAXf
            bMAXf
            bMAXc

            where the actions are defined as:
            - k ("check")
            - bMIN ("bet pot")
            - bMID ("bet 2x pot size")
            - bMAX ("bet 4x pot size")
            - c ("call")
            - f ("fold")

            This is easy calculation. If someone raises, then treat that as bMAX.

            If we raise and the opponent raises, then we treat that as bMAX. So this way, we can always
            treat the last action as bMAX.

            bMINbMAX = kBMAX
            """
            assert (
                not self._game_stage_ended()
            )  # game_stage_ended would mean that it is a chance node

            if len(self.history) == 2:
                return ["c", "bMIN", "bMID", "bMAX", "f"]
            elif self.history[-1] == "bMIN":
                return ["bMID", "bMAX", "f", "c"]
            elif self.history[-1] == "bMID":
                return ["bMAX", "f", "c"]
            elif self.history[-1] == "bMAX":
                return ["f", "c"]
            else:
                return ["k", "bMIN", "bMID", "bMAX"]

        else:
            raise Exception("Cannot call actions on a terminal history")

    def player(self):
        """
        1. ['AkTh', 'QdKd', 'bMID', 'c', '/', 'Qh2d3s4h5s']
        """
        if len(self.history) < 2:
            return -1
        elif self._game_stage_ended():
            return -1
        elif self.history[-1] == "/":
            return -1
        else:
            return (len(self.history) + 1) % 2

    def _game_stage_ended(self):
        return (
            (self.history[-1] == "c" and len(self.history) > 3)
            or self.history[-1] == "f"
            or self.history[-2:] == ["c", "k"]
        )

    def is_chance(self):
        return super().is_chance()

    def sample_chance_outcome(self):
        assert self.is_chance()

        if len(self.history) == 0:
            return "".join(player_hands[self.sample_id])
        elif len(self.history) == 1:
            return "".join(opponent_hands[self.sample_id])
        elif self.history[-1] != "/":
            return "/"
        else:
            return "".join(boards[self.sample_id])

    def terminal_utility(self, i: Player) -> int:
        assert self.is_terminal()  # We can only call the utility for a terminal history
        assert i in [0, 1]  # Only works for 2 player games for now

        winner = winners[self.sample_id]

        pot_size, _ = self._get_total_pot_size(self.history)

        if "f" in self.history:
            fold_idx = self.history.index("f")
            pot_size, latest_bet = self._get_total_pot_size(self.history[: fold_idx - 1])
            if self.history[-3] in ["bMIN", "bMID"]:  # this is part of the profit
                pot_size += latest_bet

            if len(self.history) % 2 == i:  # i is the player who folded
                return -pot_size / 2
            else:
                return pot_size / 2

        # showdown
        if winner == 0:  # tie
            return 0

        if (winner == 1 and i == 0) or (winner == -1 and i == 1):
            return pot_size / 2
        else:
            return -pot_size / 2

    def _get_total_pot_size(self, history):
        stage_total = 3  # initially 3 chips in the pot (1 SB and 1 BB)
        latest_bet = 2

        # since I am allowing
        for idx, action in enumerate(history):
            if action == "bMIN":
                old_stage_total = stage_total
                stage_total = latest_bet + stage_total  # bet ~ 1x pot
                latest_bet = old_stage_total
            elif action == "bMID":
                old_stage_total = stage_total
                stage_total = latest_bet + 2 * stage_total  # bet 2x pot
                latest_bet = 2 * old_stage_total
            elif action == "bMAX":
                old_stage_total = stage_total
                stage_total = latest_bet + 4 * stage_total  # bet all in
                latest_bet = 4 * old_stage_total
            elif action == "c":
                stage_total = 2 * latest_bet  # call

        return stage_total, latest_bet

    def __add__(self, action: Action):
        new_history = PreflopHoldemHistory(self.history + [action], self.sample_id)
        return new_history

    def get_infoSet_key(self) -> List[Action]:
        """
        This is where we abstract away cards and bet sizes.
        """
        assert not self.is_chance()
        assert not self.is_terminal()

        player = self.player()
        infoset = []
        # ------- CARD ABSTRACTION -------
        infoset.append(str(get_preflop_cluster_id(self.history[player])))
        for i, action in enumerate(self.history):
            if action in DISCRETE_ACTIONS:
                infoset.append(action)

        return infoset


class PreflopHoldemInfoSet(base.InfoSet):
    """
    Information Sets (InfoSets) cannot be chance histories, nor terminal histories.
    This condition is checked when infosets are created.

    This infoset is an abstracted versions of the history in this case.
    See the `get_infoSet_key(self)` function for these

    There are 2 abstractions we are doing:
            1. Card Abstraction (grouping together similar hands)
            2. Action Abstraction

    I've imported my abstractions from `abstraction.py`.

    """

    def __init__(self, infoSet_key: List[Action], actions: List[Action], player: Player):
        assert len(infoSet_key) >= 1
        super().__init__(infoSet_key, actions, player)


def create_infoSet(infoSet_key: List[Action], actions: List[Action], player: Player):
    """
    We create an information set from a history.
    """
    return PreflopHoldemInfoSet(infoSet_key, actions, player)


def create_history(sample_id):
    return PreflopHoldemHistory(sample_id=sample_id)


class PreflopHoldemCFR(base.CFR):
    def __init__(
        self,
        create_infoSet,
        create_history,
        n_players: int = 2,
        iterations: int = 1000000,
    ):
        super().__init__(create_infoSet, create_history, n_players, iterations)


def evaluate_winner(board, player_hand, opponent_hand):
    p1_score = evaluate_cards(*(board + player_hand))
    p2_score = evaluate_cards(*(board + opponent_hand))
    if p1_score < p2_score:
        return 1
    elif p1_score > p2_score:
        return -1
    else:
        return 0



if __name__ == "__main__":
    # Train in batches of 50,000 hands
    ITERATIONS = 50000
    cfr = PreflopHoldemCFR(create_infoSet, create_history, iterations=ITERATIONS)
    for i in range(20):
        try:
            abstraction.load_dataset(i)
        except Exception as e:
            print("Got error loading dataset: ", e)
            print("Generating new dataset")
            abstraction.generate_dataset(batch=i)

        # ----- Load the variables locally -----
        boards = abstraction.boards
        player_hands = abstraction.player_hands
        opponent_hands = abstraction.opponent_hands
        winners = abstraction.winners

        cfr.solve(debug=False, method="vanilla")
        cfr.export_infoSets(f"preflop_infoSets_batch_{i}.joblib")
