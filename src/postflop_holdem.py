"""
Abstracted version of No Limit Texas Hold'Em Poker for post-flop onwards. Also see `preflop_holdem.py` for the preflop version.

I do this to make it computationally feasible to solve on my macbook.


Card Abstraction (equity only)
- 10 clusters for flop
- 10 clusters for turn
- 10 clusters for river

Total = 10^3 = 1000 clusters

TODO: More refined card abstraction using equity distribution (though this probs make convergence take longer for infosets)
Card Abstraction (equity distribution, to compute potential of hand)
- 50 clusters for flop
- 50 clusters for turn
- 10 clusters for river (this only needs equity)

Total = 10 * 50^2 = 25000 clusters

Bet abstraction (ONLY allow these 11 sequences), more on these below
- kk
- kbMINf
- kbMINc
- kbMAXf
- kbMAXc
- bMINf
- bMINc
- bMINbMAXf
- bMINbMAXc
- bMAXf
- bMAXc

we get 11^3 = 1331 possible sequences (3 betting rounds: flop, turn, river)

In total, we have 1000 * 1331 = 1 331 000 information sets.
I noticed that only ~10% of the information sets are actually visited, so
we end up with only ~133 100 information sets.

This keeps it manageable for training.
"""

import base
from base import Player, Action
from typing import List
from abstraction import predict_cluster
import abstraction

DISCRETE_ACTIONS = ["k", "bMIN", "bMAX", "c", "f"]


class PostflopHoldemHistory(base.History):
    """
    Example of history:
    First two actions are the cards dealt to the players. The rest of the actions are the actions taken by the players.
            1. ['AkTh', 'QdKd', '/', 'QhJdKs', 'bMIN', 'c', '/', 'Ah', 'k', 'k', ...]

    Notice that there are no bets on the preflop, as this is the postflop version of the game.

    Infoset:
    [4, 'bMIN', 'c', '10', 'k', 'k', ...]


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
        self.sample_id = sample_id
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
        """
        To limit this game going to infinity, I only allow 11 betting seqeunces.
        Else the branching factor huge.
        - kk
        - kbMINf
        - kbMINc
        - kbMAXf
        - kbMAXc
        - bMINf
        - bMINc
        - bMINbMAXf
        - bMINbMAXc
        - bMAXf
        - bMAXc

        where the actions are defined as:
        - k ("check")
        - bMIN ("bet 1/3 pot, or big blind if pot is too")
        - bMAX ("bet the pot size")
        - c ("call")
        - f ("fold")

        For deeper bet sequences, this can be abstracted by collapsing the betting sequence to one of the shorter 11 sequences
        above. For example, if we raise and the opponent raises, and we raise again (ex: b100b200b300), then we treat that as simply bMAX.

        bMINbMAX = kBMAX
        """
        if self.is_chance():  # draw cards
            return (
                []
            )  # This should return the entire deck with current cards removed, but I do this for speedup by loading an existing dataset

        elif not self.is_terminal():

            assert (
                not self._game_stage_ended()
            )  # game_stage_ended would mean that it is a chance node

            if self.history[-1] == "k":
                return ["k", "bMIN", "bMAX"]
            elif self.history[-2:] == ["k", "bMIN"]:
                return ["f", "c"]
            elif self.history[-1] == "bMIN":
                return ["bMAX", "f", "c"]
            elif self.history[-1] == "bMAX":
                return ["f", "c"]
            else:
                return ["k", "bMIN", "bMAX"]

        else:
            raise Exception("Cannot call actions on a terminal history")

    def player(self):
        """
        # non dealer, dealer
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

        pot_size, _ = self._get_total_pot_size(self.history)

        last_game_stage = self.get_last_game_stage()

        if self.history[-1] == "f":
            pot_size, latest_bet = self._get_total_pot_size(self.history[:-2])
            if self.history[-3] == "bMIN":
                pot_size += latest_bet  # This is needed to calculate the correct profit

            if (
                len(last_game_stage) % 2 == i
            ):  # this isn't perfectly exact, but it's an approximation
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
        latest_bet = 0

        # note that this logic works, because I don't allow multiple raises
        for idx, action in enumerate(history):
            if action == "/":
                total += stage_total
                stage_total = 0
                latest_bet = 0  # reset latest bet in new stage
            elif action == "bMIN":
                latest_bet = max(2, int(total / 3))  # bet 1/3 pot
                stage_total += latest_bet
            elif action == "bMAX":
                latest_bet = total  # bet the pot
                stage_total += latest_bet
            elif action == "c":
                stage_total = 2 * latest_bet

        total += stage_total
        return total, latest_bet

    def __add__(self, action: Action):
        new_history = PostflopHoldemHistory(self.history + [action], self.sample_id)
        return new_history

    def get_infoSet_key_online(self) -> List[Action]:
        history = self.history
        player = self.player()
        infoset = []
        # ------- CARD ABSTRACTION -------
        # Assign cluster ID for FLOP/TURN/RIVER
        stage_i = 0
        hand = []
        if player == 0:
            hand = [history[0][:2], history[0][2:4]]
        else:
            hand = [history[1][:2], history[1][2:4]]
        community_cards = []
        for i, action in enumerate(history):
            if action not in DISCRETE_ACTIONS:
                if action == "/":
                    stage_i += 1
                    continue
                if stage_i != 0:
                    community_cards += [history[i][j : j + 2] for j in range(0, len(action), 2)]
                if stage_i == 1:
                    assert len(action) == 6
                    infoset.append(str(predict_cluster(hand + community_cards)))
                elif stage_i == 2:
                    assert len(action) == 2
                    infoset.append(str(predict_cluster(hand + community_cards)))
                elif stage_i == 3:
                    assert len(action) == 2
                    infoset.append(str(predict_cluster(hand + community_cards)))
            else:
                infoset.append(action)

        print("my hand with community cards: ", hand + community_cards)
        return "".join(infoset)

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


class PostflopHoldemInfoSet(base.InfoSet):
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
    return PostflopHoldemInfoSet(infoSet_key, actions, player)


def create_history(sample_id):
    return PostflopHoldemHistory(sample_id=sample_id)


class PostflopHoldemCFR(base.CFR):
    def __init__(
        self,
        create_infoSet,
        create_history,
        n_players: int = 2,
        iterations: int = 1000000,
    ):
        super().__init__(create_infoSet, create_history, n_players, iterations)


if __name__ == "__main__":
    # Train in batches of 50,000 hands
    ITERATIONS = 50000
    cfr = PostflopHoldemCFR(create_infoSet, create_history, iterations=ITERATIONS)
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
        player_flop_clusters = abstraction.player_flop_clusters
        player_turn_clusters = abstraction.player_turn_clusters
        player_river_clusters = abstraction.player_river_clusters
        opp_flop_clusters = abstraction.opp_flop_clusters
        opp_turn_clusters = abstraction.opp_turn_clusters
        opp_river_clusters = abstraction.opp_river_clusters
        winners = abstraction.winners

        print(boards[0])
        cfr.solve(debug=False, method="vanilla")
        cfr.export_infoSets(f"postflop_infoSets_batch_{i}.joblib")
