import base
import numpy as np
from base import Player, Action
from tqdm import tqdm
from typing import NewType, Dict, List, Callable, cast
import copy
from fast_evaluator import Deck
from abstraction import (
    get_preflop_cluster_id,
    predict_cluster,
    predict_cluster_fast,
    load_kmeans_classifiers,
)
from fast_evaluator import phEvaluatorSetup, evaluate_cards
import time


# ----- GLOBAL VARIABLES Load the pre-generated dataset -----
def load_dataset():
    global boards, player_hands, opponent_hands
    global player_preflop_clusters, player_flop_clusters, player_turn_clusters, player_river_clusters
    global opp_preflop_clusters, opp_flop_clusters, opp_turn_clusters, opp_river_clusters
    global winners

    # Load the pre-generated dataset
    boards = np.load("dataset/boards.npy").tolist()
    player_hands = np.load("dataset/player_hands.npy").tolist()
    opponent_hands = np.load("dataset/opponent_hands.npy").tolist()

    # Load player clusters
    player_preflop_clusters = np.load("dataset/player_preflop_clusters.npy").tolist()
    player_flop_clusters = np.load("dataset/player_flop_clusters.npy").tolist()
    player_turn_clusters = np.load("dataset/player_turn_clusters.npy").tolist()
    player_river_clusters = np.load("dataset/player_river_clusters.npy").tolist()

    # Load opponent clusters
    opp_preflop_clusters = np.load("dataset/opp_preflop_clusters.npy").tolist()
    opp_flop_clusters = np.load("dataset/opp_flop_clusters.npy").tolist()
    opp_turn_clusters = np.load("dataset/opp_turn_clusters.npy").tolist()
    opp_river_clusters = np.load("dataset/opp_river_clusters.npy").tolist()

    winners = np.load("dataset/winners.npy")


class HoldEmHistory(base.History):
    """
    Example of history:
    First two actions are the cards dealt to the players. The rest of the actions are the actions taken by the players.
            1. ['AkTh', 'QdKd', 'b2', 'c', '/', 'QhJdKs', 'b2', 'c', '/', 'Kh', 'k', 'k', ...]

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

    The API for the history is inspired from the Slumbot API.

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
        is_showdown = self.history.count("/") == 3 and (
            self.history[-1] == "c"  # call
            or self.history[-2:] == ["k", "k"]  # check,check
            or self._get_total_pot_size() == 200  # all-in
        )  # Showdown, since one of the players is calling
        if folded or is_showdown:
            return True
        else:
            return False

    def actions(self):
        if self.is_chance():  # draw cards
            if len(self.history) > 2 and self.history[-1] != "/":
                return ["/"]
            else:
                # cards_to_exclude = self._get_current_cards()
                # cards = Deck(cards_to_exclude)
                # return cards
                return (
                    []
                )  # This should return the entire deck with current cards removed, but I do this for speedup by loading an existing dataset

        elif not self.is_terminal():
            assert (
                not self._game_stage_ended()
            )  # game_stage_ended would mean that it is a chance node
            """
            To limit this game going to infinity, I only allow for 3 betting rounds.
            I.e. if I bet, you raise, I raise, you raise, then I must either call, fold, or all-in. Else the branching factor is going to be insane.
            """

            actions = ["k", "c", "f"]
            player = self.player()
            remaining_amount = self._get_remaining_balance(
                player, include_curr_stage=False
            )  # this is how much they can put in this round
            opp_remaining_amount = self._get_remaining_balance(
                (player + 1) % 2, include_curr_stage=True
            )  # this is how much the opponent can put in this round
            pot_size = self._get_total_pot_size()
            min_bet = self._get_min_bet()

            if opp_remaining_amount == 0 and remaining_amount > 0:
                return ["c", "f"]

            current_game_stage_history, stage = self._get_current_game_stage_history()

            # ------ BET ABSTRACTION ------
            # doing the abstraction here because the number of actions is too large, potential raise values
            # for preflop, 4 choices: check, call, 2x pot, all-in, fold
            # for flop 5 choices: check, 1/2 pot, pot, 2x pot, all-in, fold
            # For turn, 5 choices:  check, 1/2 pot, pot, 2x pot, all-in, fold
            # For river, 5 choices:  check, 1/2 pot, pot, 2x pot, all-in, fold

            # Abstract away the actions, since there are too many of them
            # history = infoSet_key

            if len(current_game_stage_history) > 3:  # prevent the game from going on forever
                return ["c", "f", "b" + str(remaining_amount)]  # all-in

            if (
                stage != "preflop"
                and int(0.5 * pot_size) < remaining_amount
                and int(0.5 * pot_size) >= min_bet
            ):  # 1/2 pot
                actions.append("b" + str(int(0.5 * pot_size)))
            if stage != "preflop" and pot_size < remaining_amount:  # pot
                actions.append("b" + str(pot_size))
            if 2 * pot_size < remaining_amount:  # 2x pot
                actions.append("b" + str(2 * pot_size))
            actions.append("b" + str(remaining_amount))  # all-in

            # Pre-flop
            if stage == "preflop":
                # Small blind to act
                if (
                    len(current_game_stage_history) == 0
                ):  # Action on SB (Dealer), who can either call, bet, or fold
                    actions.remove("k")  # You cannot check
                    return actions

                # big blind to act
                elif len(current_game_stage_history) == 1:  # 2-bet
                    if (
                        current_game_stage_history[0] == "c"
                    ):  # Small blind called, you don't need to fold, but you also can't call
                        actions.remove("f")
                        actions.remove("c")
                        return actions
                    else:  # Other player has bet, so you cannot check
                        actions.remove("k")
                        return actions
                else:
                    actions.remove("k")

                # elif len(current_game_stage_history) == 2: # 3-bet
                # 	# You cannot check at this point
                # 	actions = ['b1', 'all-in', 'c', 'f']

                # elif len(current_game_stage_history) == 3: # 4-bet
                # 	actions = ['all-in', 'c', 'f']

            else:  # flop, turn, river
                if len(current_game_stage_history) == 0:
                    actions.remove("f")  # You cannot fold
                    actions.remove("c")  # You cannot call
                elif len(current_game_stage_history) == 1:
                    if current_game_stage_history[0] == "k":
                        actions.remove("f")
                        actions.remove("c")
                    else:  # Opponent has bet, so you cannot check
                        actions.remove("k")
                else:
                    actions.remove("k")

            return actions
        else:
            raise Exception("Cannot call actions on a terminal history")

    def player(self):
        """
        This part is confusing for heads-up no limit poker, because the player that acts first changes:
        The Small Blind (SB) acts first pre-flop, but the Big Blind (BB) acts first post-flop. (see https://en.wikipedia.org/wiki/Texas_hold_%27em)
        1. ['AkTh', 'QdKd', 'b2', 'c', '/', 'Qh', 'b2', 'c', '/', '2d', b2', 'f']
                                                 SB	   BB		 	   BB	 SB 	   		BB 	 SB
        """
        if len(self.history) <= 1:
            return -1
        elif self._game_stage_ended():
            return -1
        elif self.history[-1] == "/":
            return -1
        else:
            if "/" in self.history:
                return (len(self.history) + 1) % 2  # Order is flipped post-flop
            else:
                return len(self.history) % 2

    def is_chance(self):
        return super().is_chance()

    def sample_chance_outcome(self):
        assert self.is_chance()

        # slow way, sampling manually without abstractions
        # cards = self.actions()  # Will be either or cards not seen in the deck or ['/']
        # if len(self.history) <= 1:  # We need to deal two cards to each player
        #     cards = random.sample(cards, 2)
        #     return "".join(cards)
        # else:
        #     return random.choice(cards)  # Sample one of the community cards with equal probability

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

    def terminal_utility(self, i: Player) -> int:
        assert self.is_terminal()  # We can only call the utility for a terminal history
        assert i in [0, 1]  # Only works for 2 player games for now
        winner = winners[self.sample_id]
        if winner == 0:  # tie
            return 0

        pot_size = self._get_total_pot_size()

        if winner == i:
            return pot_size
        else:
            return -pot_size

    def _get_current_cards(self):
        current_cards = []
        new_stage = False
        stage_i = 0
        for i, action in enumerate(self.history):
            if new_stage:
                new_stage = False
                if stage_i == 1:  # Flop, so there are 3 community cards
                    assert len(action) == 6
                    current_cards.append(action[:2])  # Community card 1
                    current_cards.append(action[2:4])  # Community card 2
                    current_cards.append(action[4:6])  # Community card 3

                else:  # Turn or river
                    current_cards.append(action)  # Community card
            elif action == "/":
                new_stage = True
                stage_i += 1

            elif i == 0 or i == 1:
                assert len(action) == 4
                current_cards.append(action[:2])  # Private card 1
                current_cards.append(action[2:4])  # Private card 2

        return current_cards

    def _get_current_game_stage_history(self):
        """
        return current_game_stage_history, stages[stage_i] excluding the community cards drawn. We only care about the actions
        of the players.
        """
        game_stage_start = 2  # Because we are skipping the pairs of private cards drawn at the beginning of the round
        stage_i = 0
        stages = ["preflop", "flop", "turn", "river"]
        for i, action in enumerate(self.history):
            if action == "/":
                game_stage_start = i + 2  # Skip the community card
                stage_i += 1

        if game_stage_start >= len(self.history):
            return [], stages[stage_i]
        else:
            current_game_stage_history = self.history[game_stage_start:]
            return current_game_stage_history, stages[stage_i]

    def _get_min_bet(self):
        # TODO: Test this function
        curr_bet = 0
        prev_bet = 0
        for i in range(len(self.history) - 1, 0, -1):
            if self.history[i][0] == "b":  # Bet, might be a raise
                if curr_bet == 0:
                    curr_bet = int(self.history[i][1:])
                elif prev_bet == 0:
                    prev_bet = int(self.history[i][1:])
            elif self.history[i] == "/":
                break

        # Handle case when game stage is preflop, in which case a bet is already placed for you
        game_stage_history, game_stage = self._get_current_game_stage_history()
        if game_stage == "preflop" and curr_bet == 0:
            curr_bet = 2  # big blind
        elif curr_bet == 0:  # No bets has been placed
            assert prev_bet == 0
            curr_bet = 1

        return int(curr_bet + (curr_bet - prev_bet))  # This is the minimum raise

    def _calculate_player_total(self, player: Player, include_curr_stage=False):
        """
        This is the amount of money a player has put into the pot, INCLUDING the current game stage.

        In preflop, this is 0.
        """
        stage_i = 0
        # Total across all game stages (preflop, flop, turn, river)
        # initial values for big and small blind
        player_total = 0
        if player == 0:
            player_game_stage_total = 1
        else:
            player_game_stage_total = 2

        i = 0
        for hist_idx, hist in enumerate(self.history):
            if i == player:
                if hist[0] == "b":
                    player_game_stage_total = int(hist[1:])
                elif hist == "k":
                    if stage_i == 0:  # preflop, checking means 2
                        player_game_stage_total = 2
                    else:
                        player_game_stage_total = 0
                elif hist == "c":  # Call the previous bet
                    # Exception for when you can call the big blind on the preflop, without the big blind having bet previously
                    if hist_idx == 2:
                        player_game_stage_total = 2
                    else:
                        player_game_stage_total = int(self.history[hist_idx - 1][1:])

            i = (i + 1) % 2

            if hist == "/":
                stage_i += 1
                player_total += player_game_stage_total
                player_game_stage_total = 0
                if stage_i == 1:
                    i = (
                        i + 1
                    ) % 2  # We need to flip the order post-flop, as the BB is the one who acts first now

        if include_curr_stage:
            player_total += player_game_stage_total
        return player_total

    def _get_remaining_balance(self, player: Player, include_curr_stage=False):
        # Each player starts with a balance of 100 at the beginning of each hand
        return 100 - self._calculate_player_total(player, include_curr_stage=include_curr_stage)

    def _get_stage_pot_size(self):
        game_stage_history, stage = self._get_current_game_stage_history()
        max_player_bet = 0
        max_opp_bet = 0
        if stage == "preflop":
            max_player_bet = 1
            max_opp_bet = 2
            if "c" in game_stage_history:
                max_player_bet = 2

        for i, action in enumerate(game_stage_history):
            if action[0] == "b":
                if i % 2 == 0:
                    max_player_bet = int(action[1:])
                else:
                    max_opp_bet = int(action[1:])

        pot_size = max_player_bet + max_opp_bet
        return pot_size

    def _get_total_pot_size(self):
        return +self._calculate_player_total(
            0, include_curr_stage=True
        ) + self._calculate_player_total(1, include_curr_stage=True)

    def _game_stage_ended(self):
        # TODO: Make sure this logic is good
        current_game_stage_history, stage = self._get_current_game_stage_history()
        if len(current_game_stage_history) == 0:
            return False
        elif current_game_stage_history[-1] == "f":
            return True
        elif (
            current_game_stage_history[-1] == "c" and len(self.history) > 3
        ):  # On pre-flop, when the small blind calls, the opponent can still bet
            return True
        elif stage == "preflop" and current_game_stage_history[-2:] == ["c", "k"]:
            return True
        elif len(current_game_stage_history) >= 2 and current_game_stage_history[-2:] == [
            "k",
            "k",
        ]:  # check, check
            return True
        else:
            # both players all-in, with a call (IMPORTANT ASSUMPTION: both players starting balance is the same)
            remaining_amount = self._get_remaining_balance(0, include_curr_stage=True)
            opp_remaining_amount = self._get_remaining_balance(1, include_curr_stage=True)
            if remaining_amount == 0 and opp_remaining_amount == 0:
                return True
            return False

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
        history = copy.deepcopy(self.history)

        # ------- CARD ABSTRACTION -------
        # Assign cluster ID for PREFLOP cards
        if player == 0:
            history[0] = str(player_preflop_clusters[self.sample_id])
            history[1] = "?"
        else:
            history[0] = "?"
            history[1] = str(opp_preflop_clusters[self.sample_id])

        # Assign cluster ID for FLOP/TURN/RIVER
        new_stage = False
        stage_i = 0
        for i, action in enumerate(history):
            if new_stage:
                new_stage = False
                if stage_i == 1:
                    assert len(action) == 6
                    if player == 0:
                        history[i] = str(player_flop_clusters[self.sample_id])
                    else:
                        history[i] = str(opp_flop_clusters[self.sample_id])

                elif stage_i == 2:
                    assert len(action) == 2
                    if player == 0:
                        history[i] = str(player_turn_clusters[self.sample_id])
                    else:
                        history[i] = str(opp_turn_clusters[self.sample_id])
                elif stage_i == 3:
                    assert len(action) == 2
                    if player == 0:
                        history[i] = str(player_river_clusters[self.sample_id])
                    else:
                        history[i] = str(opp_river_clusters[self.sample_id])
            elif action == "/":
                new_stage = True
                stage_i += 1

        return history


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
        assert len(infoSet_key) >= 2
        abstracted_actions = copy.deepcopy(actions)
        super().__init__(infoSet_key, abstracted_actions, player)


def create_infoSet(infoSet_key: List[Action], actions: List[Action], player: Player):
    """
    We create an information set from a history.
    """
    return HoldemInfoSet(infoSet_key, actions, player)


def create_history(sample_id):
    return HoldEmHistory(sample_id=sample_id)


class HoldemCFR(base.CFR):
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


def generate_dataset(iterations=1000, num_samples=10000, save=True):
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

    # player_preflop_clusters = Parallel(n_jobs=-1)(
    #     delayed(get_preflop_cluster_id)(cards) for cards in player_hands
    # )
    player_preflop_clusters = Parallel(n_jobs=-1)(
        delayed(predict_cluster_fast)(cards, n=3000, total_clusters=20)
        for cards in tqdm(player_hands)
    )
    player_flop_clusters = Parallel(n_jobs=-1)(
        delayed(predict_cluster_fast)(cards, n=1000, total_clusters=10)
        for cards in tqdm(player_flop_cards)
    )
    player_turn_clusters = Parallel(n_jobs=-1)(
        delayed(predict_cluster_fast)(cards, n=500, total_clusters=5)
        for cards in tqdm(player_turn_cards)
    )
    player_river_clusters = Parallel(n_jobs=-1)(
        delayed(predict_cluster_fast)(cards, n=200, total_clusters=5)
        for cards in tqdm(player_river_cards)
    )

    opp_preflop_clusters = Parallel(n_jobs=-1)(
        delayed(predict_cluster_fast)(cards, n=3000, total_clusters=20)
        for cards in tqdm(opponent_hands)
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

    print("saving datasets")
    np.save("dataset/boards.npy", boards)
    np.save("dataset/player_hands.npy", player_hands)
    np.save("dataset/opponent_hands.npy", opponent_hands)
    np.save("dataset/winners.npy", winners)
    print("continuing to save datasets")

    np.save("dataset/player_preflop_clusters.npy", player_preflop_clusters)
    np.save("dataset/player_flop_clusters.npy", player_flop_clusters)
    np.save("dataset/player_turn_clusters.npy", player_turn_clusters)
    np.save("dataset/player_river_clusters.npy", player_river_clusters)

    np.save("dataset/opp_preflop_clusters.npy", opp_preflop_clusters)
    np.save("dataset/opp_flop_clusters.npy", opp_flop_clusters)
    np.save("dataset/opp_turn_clusters.npy", opp_turn_clusters)
    np.save("dataset/opp_river_clusters.npy", opp_river_clusters)

    print(time.time() - curr)


import joblib

if __name__ == "__main__":
    # generate_dataset()
    load_dataset()
    cfr = HoldemCFR(create_infoSet, create_history)
    # cfr.infoSets = joblib.load("infosets.joblib")
    cfr.solve()

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
