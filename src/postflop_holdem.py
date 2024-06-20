"""
Abstracted version of Holdem Poker, used for training.

To make this computationally feasible to solve on my macbook, I start solving at the flop.

Card Abstraction
- 10 clusters for flop
- 5 clusters for turn
- 5 clusters for river

10 * 5 * 5 = 250 clusters

Bet abstraction (ONLY allow these 11 sequences)
- k ("check")
- bMIN ("bet 1/3 pot, or big blind if pot is too")
- bMAX ("bet the pot size")
- c ("call")
- f ("fold")

kk
kbMINf
kbMINc
kbPOTf
kbPOTc
bMINf
bMINc
bMINbMAXf # opponent raises on you
bMINbMAXc # opponent raises on you
bPOTf
bPOTc

11^3 = 1331 possible sequences (3 stages: flop, turn, river)

In total, we have 250 * 1331 = 332750 information sets.

This keeps it manageable. Anything more is in orders of millions...
"""

import base
import numpy as np
from base import Player, Action
from tqdm import tqdm
from typing import List
from abstraction import (
    predict_cluster_pre,
)
from fast_evaluator import phEvaluatorSetup, evaluate_cards
import time

DISCRETE_ACTIONS = ["k", "bMIN", "bMAX", "c", "f"]


# ----- GLOBAL VARIABLES Load the pre-generated dataset -----
def load_dataset():
    global boards, player_hands, opponent_hands
    global player_flop_clusters, player_turn_clusters, player_river_clusters
    global opp_flop_clusters, opp_turn_clusters, opp_river_clusters
    global winners

    # Load the pre-generated dataset
    boards = np.load("dataset/boards.npy").tolist()
    player_hands = np.load("dataset/player_hands.npy").tolist()
    opponent_hands = np.load("dataset/opponent_hands.npy").tolist()

    # Load player clusters
    player_flop_clusters = np.load("dataset/player_flop_clusters.npy").tolist()
    player_turn_clusters = np.load("dataset/player_turn_clusters.npy").tolist()
    player_river_clusters = np.load("dataset/player_river_clusters.npy").tolist()

    # Load opponent clusters
    opp_flop_clusters = np.load("dataset/opp_flop_clusters.npy").tolist()
    opp_turn_clusters = np.load("dataset/opp_turn_clusters.npy").tolist()
    opp_river_clusters = np.load("dataset/opp_river_clusters.npy").tolist()

    winners = np.load("dataset/winners.npy")


class HoldEmHistory(base.History):
    """
    Example of history:
    First two actions are the cards dealt to the players. The rest of the actions are the actions taken by the players.
            1. ['AkTh', 'QdKd', '/', 'QhJdKs', 'bMIN', 'c', '/', 'Ah', 'k', 'k', ...]

    Infoset:
    [4, 'bMIN', 'c', '10', 'k', 'k', ...]

    ---- ACTIONS ----
    - k = check
    - bX = bet X amount (this includes raising)
    - c = call
    - f = fold (you cannot fold if the other player just checked)

    Every round starts the same way:
    Small blind = 1 chip
    Big blind = 2 chips

    Total chips = 100BB per player.
    Minimum raise = X to match bet, and Y is the raise amount
    If no raise before, then the minimum raise amount is 2x the bet amount (preflop would be 2x big blind).
    Else it is whatever was previously raised. This is not the same as 2x the previous bet amount. Just the Y raise amount.

    Ex: The bet is 10$. I raise to 50$, so I raised by 40$ (Y = 40). The next player's minimum raise is not 100$, but rather to 90$, since (it's 50$ to match the bet, and 40$ to match the raise).

    Minimum bet = 1 chip (0.5BB)

    The API for the history is inspired from the Slumbot API, https://www.slumbot.com/

    I want to avoid all the extra overhead, so taking inspiration from `environment.py` with the `PokerEnvironment`
    """

    def __init__(self, history: List[Action] = [], sample_id=0):
        super().__init__(history)
        self.sample_id = sample_id % len(player_hands)
        self.stage_i = history.count("/")

    def is_terminal(self):
        if len(self.history) == 0:
            return False
        folded = self.history[-1] == "f"
        is_showdown = self.stage_i == 3 and self._game_stage_ended()  # call  # check,check
        if folded or is_showdown:
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
            To limit this game going to infinity, I only allow 11 betting seqeunces.
            Else the branching factor huge.

            kk
            kbMINf
            kbMINc
            kbMAXf
            kbMAXc
            bMINf
            bMINc
            bMINbMAXf
            bMINbMAXc
            bMAXf
            bMAXc

            This is easy calculation. If someone raises, then treat that as bMAX.

            If we raise and the opponent raises, then we treat that as bMAX. So this way, we can always
            treat the last action as bMAX.

            bMINbMAX = kBMAX
            """
            assert (
                not self._game_stage_ended()
            )  # game_stage_ended would mean that it is a chance node

            if self.history[-1] == "k":
                return ["k", "bMIN", "bMAX"]
            elif self.history[-1] == "bMIN":
                return ["f", "c"]
            elif self.history[-1] == "bMAX":
                return ["f", "c"]
            else:
                return ["k", "bMIN", "bMAX"]

        else:
            raise Exception("Cannot call actions on a terminal history")

    def player(self):
        """
        1. ['AkTh', 'QdKd', '/', 'Qh', 'b2', 'c', '/', '2d', b2', 'f']
        """
        if len(self.history) <= 3:
            return -1
        elif self._game_stage_ended():
            return -1
        elif self.history[-1] == "/":
            return -1
        else:
            last_game_stage = self.get_last_game_stage()
            # latest game stage
            return (len(last_game_stage) + 1) % 2

    def _game_stage_ended(self):
        return self.history[-1] == "c" or self.history[-1] == "f" or self.history[-2:] == ["k", "k"]

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
        elif self.stage_i == 1:
            return "".join(boards[self.sample_id][:3])
        elif self.stage_i == 2:
            return boards[self.sample_id][3]
        elif self.stage_i == 3:
            return boards[self.sample_id][4]

    def get_last_game_stage(self):
        last_game_stage_start_idx = max(loc for loc, val in enumerate(self.history) if val == "/")
        last_game_stage = self.history[last_game_stage_start_idx + 1 :]
        return last_game_stage

    def terminal_utility(self, i: Player) -> int:
        assert self.is_terminal()  # We can only call the utility for a terminal history
        assert i in [0, 1]  # Only works for 2 player games for now

        winner = winners[self.sample_id]

        pot_size = self._get_total_pot_size(self.history)

        last_game_stage = self.get_last_game_stage()

        if self.history[-1] == "f":
            pot_size = self._get_total_pot_size(self.history[:-2])
            if len(last_game_stage) % 2 == i:
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
        total = 0
        stage_total = 4  # assume preflop is a check + call, so 4 in pot (1 BB = 2 chips)
        for idx, action in enumerate(history):
            if action == "/":
                total += stage_total
                stage_total = 0
            elif action == "bMIN":
                stage_total += max(2, int(total / 3))  # bet 1/3 pot
            elif action == "bMAX":
                stage_total += total  # bet the pot
            elif action == "c":
                if history[idx - 1] == "bMIN":
                    stage_total += max(2, int(total / 3))
                elif history[idx - 1] == "bMAX" and history[idx - 2] == "bMIN":
                    stage_total = 2 * total
                elif history[idx - 1] == "bMAX":
                    stage_total += total

        total += stage_total
        return total

    def __add__(self, action: Action):
        new_history = HoldEmHistory(self.history + [action], self.sample_id)
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
        # Assign cluster ID for FLOP/TURN/RIVER
        stage_i = 0
        for i, action in enumerate(self.history):
            if action not in DISCRETE_ACTIONS:
                if action == "/":
                    stage_i += 1
                    continue
                if stage_i == 1:
                    if player == 0:
                        infoset.append(str(player_flop_clusters[self.sample_id]))
                    else:
                        infoset.append(str(opp_flop_clusters[self.sample_id]))
                elif stage_i == 2:
                    assert len(action) == 2
                    if player == 0:
                        infoset.append(str(player_turn_clusters[self.sample_id]))
                    else:
                        infoset.append(str(opp_turn_clusters[self.sample_id]))
                elif stage_i == 3:
                    assert len(action) == 2
                    if player == 0:
                        infoset.append(str(player_river_clusters[self.sample_id]))
                    else:
                        infoset.append(str(opp_river_clusters[self.sample_id]))
            else:
                infoset.append(action)

        return infoset


class HoldemInfoSet(base.InfoSet):
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
    return HoldemInfoSet(infoSet_key, actions, player)


def create_history(sample_id):
    return HoldEmHistory(sample_id=sample_id)


class PostFlopHoldemCFR(base.CFR):
    def __init__(
        self,
        create_infoSet,
        create_history,
        n_players: int = 2,
        iterations: int = 1000000,
    ):
        super().__init__(create_infoSet, create_history, n_players, iterations)


from joblib import Parallel, delayed


def evaluate_winner(board, player_hand, opponent_hand):
    p1_score = evaluate_cards(*(board + player_hand))
    p2_score = evaluate_cards(*(board + opponent_hand))
    if p1_score < p2_score:
        return 1
    elif p1_score > p2_score:
        return -1
    else:
        return 0


def generate_dataset(num_samples=50000, save=True):
    """
    To make things faster, we pre-generate the boards and hands. We also pre-cluster the hands
    """
    boards, player_hands, opponent_hands = phEvaluatorSetup(num_samples)

    np_boards = np.array(boards)
    np_player_hands = np.array(player_hands)
    np_opponent_hands = np.array(opponent_hands)

    player_flop_cards = np.concatenate((np_player_hands, np_boards[:, :3]), axis=1).tolist()
    player_turn_cards = np.concatenate((np_player_hands, np_boards[:, :4]), axis=1).tolist()
    player_river_cards = np.concatenate((np_player_hands, np_boards), axis=1).tolist()
    opp_flop_cards = np.concatenate((np_opponent_hands, np_boards[:, :3]), axis=1).tolist()
    opp_turn_cards = np.concatenate((np_opponent_hands, np_boards[:, :4]), axis=1).tolist()
    opp_river_cards = np.concatenate((np_opponent_hands, np_boards), axis=1).tolist()

    curr = time.time()
    print("generating clusters")

    player_flop_clusters = Parallel(n_jobs=-1)(
        delayed(predict_cluster_fast)(cards, n=1000, total_clusters=10)
        for cards in tqdm(player_flop_cards)
    )
    player_turn_clusters = Parallel(n_jobs=-1)(
        delayed(predict_cluster_fast)(cards, n=1000, total_clusters=5)
        for cards in tqdm(player_turn_cards)
    )
    player_river_clusters = Parallel(n_jobs=-1)(
        delayed(predict_cluster_fast)(cards, n=1000, total_clusters=5)
        for cards in tqdm(player_river_cards)
    )

    opp_flop_clusters = Parallel(n_jobs=-1)(
        delayed(predict_cluster_fast)(cards, n=1000, total_clusters=10)
        for cards in tqdm(opp_flop_cards)
    )
    opp_turn_clusters = Parallel(n_jobs=-1)(
        delayed(predict_cluster_fast)(cards, n=500, total_clusters=5)
        for cards in tqdm(opp_turn_cards)
    )
    opp_river_clusters = Parallel(n_jobs=-1)(
        delayed(predict_cluster_fast)(cards, n=200, total_clusters=5)
        for cards in tqdm(opp_river_cards)
    )

    winners = Parallel(n_jobs=-1)(
        delayed(evaluate_winner)(board, player_hand, opponent_hand)
        for board, player_hand, opponent_hand in tqdm(zip(boards, player_hands, opponent_hands))
    )

    if save:
        print("saving datasets")

        np.save("dataset/boards.npy", boards)
        np.save("dataset/player_hands.npy", player_hands)
        np.save("dataset/opponent_hands.npy", opponent_hands)
        np.save("dataset/winners.npy", winners)
        print("continuing to save datasets")

        np.save("dataset/player_flop_clusters.npy", player_flop_clusters)
        np.save("dataset/player_turn_clusters.npy", player_turn_clusters)
        np.save("dataset/player_river_clusters.npy", player_river_clusters)

        np.save("dataset/opp_flop_clusters.npy", opp_flop_clusters)
        np.save("dataset/opp_turn_clusters.npy", opp_turn_clusters)
        np.save("dataset/opp_river_clusters.npy", opp_river_clusters)

    print(time.time() - curr)


if __name__ == "__main__":
    # Train in batches of 50,000 hands
    ITERATIONS = 50000
    cfr = PostFlopHoldemCFR(create_infoSet, create_history, iterations=ITERATIONS)
    for i in range(20):
        if i == 0:
            load_dataset()
        else:
            generate_dataset(save=False, num_samples=ITERATIONS)
        cfr.solve(debug=False, method="vanilla")
        cfr.export_infoSets(f"infoSets_batch_{i}.joblib")

    # load_dataset()
    # cfr.infoSets = joblib.load("infoSets_2500.joblib")
    # print("finished loading")
    # cfr.solve(debug=True)
    # cfr.solve_multiprocess(
    #     initializer=load_dataset,
    # )

#     """
# 	When we work with these abstractions, we have two types:
# 	1. Action Abstraction
# 	2. Card Abstraction

# 	Both of these are implemented in a different way.

# 	"""

#     hist: HoldEmHistory = create_history()
#     assert hist.player() == -1
#     hist1 = hist + "AkTh"
#     assert hist1.player() == -1
#     hist2 = hist1 + "QdKd"
#     assert hist2.player() == 0
#     print(hist2.get_infoSet_key(kmeans_flop, kmeans_turn, kmeans_river))
#     hist3 = hist2 + "b2"
#     assert hist3.player() == 1
#     hist4 = hist3 + "c"
#     assert hist4.player() == -1
#     # Below are chance events, so it doesn't matter which player it is
#     hist5 = hist4 + "/"
#     assert hist5.player() == -1
#     hist6 = hist5 + "QhKsKh"
#     assert hist6.player() == 1
#     hist7 = hist6 + "b1"
#     hist8: HoldEmHistory = hist7 + "b3"
#     curr = time.time()
#     print(hist8.get_infoSet_key(kmeans_flop, kmeans_turn, kmeans_river), time.time() - curr)

#     # cfr = base.CFR(create_infoSet, create_history)
#     # cfr.solve()
