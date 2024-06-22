"""
Design decisions, OOP vs Procedural:
Certain things make natural sense to represent as objects, such as `Player` and `History`. However, what about our functions, such as
the player function, the utility function, or the function that computes and updates the strategy?

We can either treat them as methods of a class (OOP), or a separate function that takes in a class as an argument (procedural).

Let us consider the terminal_utility(), which returns the utility of a terminal history for a particular player. If we choose 
to write this function outside of the class (procedural approach), we could do something like this:

def terminal_utility(i: Player, history: History) -> float:
	if isinstance(history, KuhnHistory):
		#dosomething
	elif isinstance(history, LeducHistory):
		#dosomethingelse

While this way of writing is more consistent with the mathematical notations, the problem that if we add a new game, then we would need to add a new 
elif statement for all of our functions. With OOP, we can simply make `terminal_utility()` an abstract method of the `History` class. Childrens of `History` class 
are then  forced to define the utility function. This approach helps us easily extend to more games, and is the approach I will take below.



Strategy is stored at the infoset level.
"""

# TODO: use NumPy lookup instead of dictionary lookup for drastic speed improvement https://stackoverflow.com/questions/36652533/looking-up-large-sets-of-keys-dictionary-vs-numpy-array


from typing import NewType, Dict, List
from tqdm import tqdm
import time
import joblib
import numpy as np

CHANCE = "CHANCE_EVENT"

Player = NewType("Player", int)
Action = NewType("Action", str)


class History:
    """
    The history includes the information about the set of cards we hold.

    In our 2 player version, Player 0 always is the first to act and player 1 is the second to act.
    However, be warned that in a game such as Heads-Up NL Hold'Em, in later game stages, player 1 (big blind)
    might be first to act.
    """

    def __init__(self, history: List[Action] = []):
        self.history = history

    def is_terminal(self):
        raise NotImplementedError()

    def actions(self) -> List[Action]:
        raise NotImplementedError()

    def player(self) -> Player:
        # If chance event, return -1 as described in CFR paper
        assert not self.is_terminal()
        raise NotImplementedError()

    def is_chance(self) -> bool:
        return self.player() == -1

    def sample_chance_outcome(self) -> Action:
        # TODO: Determine how to format this API
        raise NotImplementedError()

    def terminal_utility(self, i: Player) -> int:
        assert self.is_terminal()
        assert i in [0, 1]

        raise NotImplementedError()

    def __add__(self, action: Action):
        """
        This should always be something like:

                new_history = HoldemHistory(self.history + [action])
                return new_history

        """
        raise NotImplementedError()

    def get_infoSet_key(self) -> List[Action]:
        assert not self.is_chance()  # chance history should not be infosets
        assert not self.is_terminal()

        raise NotImplementedError()

    def __repr__(self) -> str:
        return str(self.history)


class InfoSet:
    """
    Most of the infoset information (actions, player) should be inherited from the history class.

    """

    def __init__(self, infoSet_key: List[Action], actions: List[Action], player: Player):
        self.infoSet = infoSet_key
        self.__actions = actions
        self.__player = player

        self.regret = {a: 0 for a in self.actions()}
        self.strategy = {a: 0 for a in self.actions()}
        self.cumulative_strategy = {a: 0 for a in self.actions()}
        self.get_strategy()
        assert 1.0 - sum(self.strategy.values()) < 1e-6

    def __repr__(self) -> str:
        return str(self.infoSet)

    def actions(self) -> List[Action]:
        return self.__actions

    def player(self) -> Player:
        return self.__player

    def to_dict(self):
        return {
            "infoset": self.infoSet,
            "regret": self.regret,
            "cumulative_strategy": self.cumulative_strategy,
        }

    def get_strategy(self):
        """
        Updates the current strategy based on the current regret, using regret matching
        """
        regret = {a: max(r, 0) for a, r in self.regret.items()}

        regret_sum = sum(regret.values())

        if regret_sum > 0:
            self.strategy = {a: r / regret_sum for a, r in regret.items()}
        else:
            self.strategy = {a: 1 / len(self.actions()) for a in self.actions()}

    def get_average_strategy(self):
        """ """
        assert len(self.actions()) == len(
            self.cumulative_strategy
        )  # The cumulative strategy should map a probability for every action

        strategy_sum = sum(self.cumulative_strategy.values())

        if strategy_sum > 0:
            return {a: s / strategy_sum for a, s in self.cumulative_strategy.items()}
        else:
            return {a: 1 / len(self.actions()) for a in self.actions()}


class CFR:
    def __init__(
        self,
        create_infoSet,
        create_history,
        n_players: int = 2,
        iterations: int = 1000000,
    ):
        self.n_players = n_players
        self.iterations = iterations
        self.tracker_interval = int(iterations / 10)
        self.infoSets: Dict[str, InfoSet] = {}
        self.create_infoSet = create_infoSet
        self.create_history = create_history

        self.tracker = InfoSetTracker()

    def get_infoSet(self, history: History) -> InfoSet:
        infoSet_key = history.get_infoSet_key()
        actions = history.actions()
        player = history.player()

        assert type(infoSet_key) == list
        assert type(actions) == list

        infoSet_key_str = "".join(infoSet_key)
        if infoSet_key_str not in self.infoSets:
            self.infoSets[infoSet_key_str] = self.create_infoSet(infoSet_key, actions, player)

        return self.infoSets[infoSet_key_str]

    def vanilla_cfr(
        self, history: History, i: Player, t: int, pi_0: float, pi_1: float, debug=False
    ):  # Works for two players
        # Return payoff for terminal states
        if history.is_terminal():
            if debug:
                print(
                    f"history: {history.history} utility: {history.terminal_utility(i)}, player: {i}"
                )
                time.sleep(0.1)
            return history.terminal_utility(i)
        elif history.is_chance():
            a = (
                history.sample_chance_outcome()
            )  # $\sigma_c$ is simply the $f_c$ function I believe...
            return self.vanilla_cfr(
                history + a, i, t, pi_0, pi_1, debug=debug
            )  # Since it is a chance outcome, the player does not change .. TODO: Check logic for this

        infoSet = self.get_infoSet(history)
        assert infoSet.player() == history.player()

        v = 0
        va = {}

        for a in infoSet.actions():
            if history.player() == 0:
                va[a] = self.vanilla_cfr(
                    history + a, i, t, infoSet.strategy[a] * pi_0, pi_1, debug=debug
                )
            else:
                va[a] = self.vanilla_cfr(
                    history + a, i, t, pi_0, infoSet.strategy[a] * pi_1, debug=debug
                )

            v += infoSet.strategy[a] * va[a]

        if history.player() == i:
            for a in infoSet.actions():
                infoSet.regret[a] += (pi_1 if i == 0 else pi_0) * (va[a] - v)
                # Update cumulative strategy values, this will be used to calculate the average strategy at the end
                infoSet.cumulative_strategy[a] += (pi_0 if i == 0 else pi_1) * infoSet.strategy[a]

            # Update regret matching values
            infoSet.get_strategy()

        if debug:
            print("infoset", infoSet.to_dict())
            print("strategy", infoSet.strategy)

        return v

    def vanilla_cfr_speedup(self, history: History, t: int, pi_0: float, pi_1: float, debug=False):
        """
        We double the speed by updating both player values simultaneously, since this is a zero-sum game.

        The trick here to speedup is by assuming by whatever the opponent gains is
        the opposite of what we gain. Zero-sum game. However, need to make sure we always return the correct utility.

        NOTE: For some reason, doesn't work super well, the strategies are not converging as well as they should.

        """
        raise NotImplementedError()  # bad implementation, does not provide Nash equilibrium solution
        # Return payoff for terminal states
        # ['3d7c', '4cQd', '/', '7sKd9c', 'bMIN', 'f']
        if history.is_terminal():
            if debug:
                print(
                    f"utility returned: {history.terminal_utility((len(history.get_last_game_stage())) % 2)}, history: {history.history}"
                )
            return history.terminal_utility(
                (len(history.get_last_game_stage()) + 1) % 2
            )  # overfit solution for holdem
        elif history.is_chance():
            a = (
                history.sample_chance_outcome()
            )  # $\sigma_c$ is simply the $f_c$ function I believe...
            return self.vanilla_cfr_speedup(
                history + a, t, pi_0, pi_1, debug=debug
            )  # Since it is a chance outcome, the player does not change .. TODO: Check logic for this

        infoSet = self.get_infoSet(history)
        assert infoSet.player() == history.player()

        v = 0
        va = {}

        for a in infoSet.actions():
            if history.player() == 0:
                va[a] = -self.vanilla_cfr_speedup(
                    history + a, t, infoSet.strategy[a] * pi_0, pi_1, debug=debug
                )
            else:
                va[a] = -self.vanilla_cfr_speedup(
                    history + a, t, pi_0, infoSet.strategy[a] * pi_1, debug=debug
                )

            v += infoSet.strategy[a] * va[a]

        for a in infoSet.actions():
            infoSet.regret[a] += (pi_1 if history.player() == 0 else pi_0) * (va[a] - v)
            # Update cumulative strategy values, this will be used to calculate the average strategy at the end
            infoSet.cumulative_strategy[a] += (
                pi_0 if history.player() == 0 else pi_1
            ) * infoSet.strategy[a]

        # Update regret matching values
        infoSet.get_strategy()

        if debug:
            print("infoset", infoSet.to_dict())
            print("va", va)
            print("strategy", infoSet.strategy)
            time.sleep(0.1)

        return v

    def vanilla_cfr_manim(
        self,
        history: History,
        i: Player,
        t: int,
        pi_0: float,
        pi_1: float,
        histories: List[History],
    ):
        # Return payoff for terminal states
        if history.is_terminal():
            histories.append(history)
            return history.terminal_utility(i)
        elif history.is_chance():
            a = (
                history.sample_chance_outcome()
            )  # $\sigma_c$ is simply the $f_c$ function I believe...
            return self.vanilla_cfr_manim(
                history + a, i, t, pi_0, pi_1, histories
            )  # Since it is a chance outcome, the player does not change .. TODO: Check logic for this

        infoSet = self.get_infoSet(history)
        assert infoSet.player() == history.player()

        v = 0
        va = {}

        for a in infoSet.actions():
            if history.player() == 0:
                va[a] = self.vanilla_cfr_manim(
                    history + a, i, t, infoSet.strategy[a] * pi_0, pi_1, histories
                )
            else:
                va[a] = self.vanilla_cfr_manim(
                    history + a, i, t, pi_0, infoSet.strategy[a] * pi_1, histories
                )

            v += infoSet.strategy[a] * va[a]

        if history.player() == i:
            for a in infoSet.actions():
                infoSet.regret[a] += (pi_1 if i == 0 else pi_0) * (va[a] - v)
                # Update cumulative strategy values, this will be used to calculate the average strategy at the end
                infoSet.cumulative_strategy[a] += (pi_0 if i == 0 else pi_1) * infoSet.strategy[a]

            # Update regret matching values
            infoSet.get_strategy()

        return v

    def mccfr(
        self, history: History, i: Player, t: int, pi_0: float, pi_1: float, debug=False
    ):  # Works for two players
        raise NotImplementedError()

    def solve(self, method="vanilla", debug=False):
        util_0 = 0
        util_1 = 0
        if method == "manim":
            histories = []

        for t in tqdm(range(self.iterations), desc="CFR Training Loop"):
            if method == "vanilla":  # vanilla
                for player in range(
                    self.n_players
                ):  # This is the slower way, we can speed by updating both players
                    if player == 0:
                        util_0 += self.vanilla_cfr(
                            self.create_history(t), player, t, 1, 1, debug=debug
                        )
                    else:
                        util_1 += self.vanilla_cfr(
                            self.create_history(t), player, t, 1, 1, debug=debug
                        )

            elif method == "vanilla_speedup":
                util_0 += self.vanilla_cfr_speedup(self.create_history(t), t, 1, 1, debug=debug)

            elif method == "manim" and t < 10:
                for player in range(self.n_players):
                    if player == 0:
                        util_0 += self.vanilla_cfr_manim(
                            self.create_history(t), player, t, 1, 1, histories
                        )
                    else:
                        util_1 += self.vanilla_cfr_manim(
                            self.create_history(t), player, t, 1, 1, histories
                        )

                print(histories)

            if (t + 1) % self.tracker_interval == 0:
                print("Average game value player 0: ", util_0 / t)
                print("Average game value player 1: ", util_1 / t)
                if len(self.infoSets) < 100000:
                    self.tracker(self.infoSets)
                    self.tracker.pprint()

        if method == "manim":
            return histories

    def export_infoSets(self, filename="infoSets.joblib"):
        joblib.dump(self.infoSets, filename)

    def get_expected_value(
        self, history: History, player: Player, player_strategy=None, opp_strategy=None
    ):
        """
        We can compute the expected values of two strategies. If none, then we will
        play both according to the nash equilibrium strategies we computed.

        However, Getting the expected value this way is not feasible for super large games such as
        no-limit texas hold'em, which is why we can compute an approximate EV (see function below).

        This is also known as the expected payoff, or utility function of a strategy profile $u_i(\sigma)$
        """
        # the counterfactual value is simply the averaged utilities possible
        if history.is_terminal():
            return history.terminal_utility(player)
        else:
            infoSet = self.get_infoSet(history)

            if history.player() == player:
                if player_strategy is not None:
                    average_strategy = player_strategy
                else:
                    average_strategy = infoSet.get_average_strategy()
            else:
                if opp_strategy is not None:
                    average_strategy = opp_strategy
                else:
                    average_strategy = infoSet.get_average_strategy()

            ev = 0
            for idx, a in enumerate(infoSet.actions()):
                value = self.get_expected_value(history + a, player, player_strategy, opp_strategy)
                ev += average_strategy[idx] * value

            return ev

    def get_expected_value_approx(self, history: History, player: Player):
        # Getting the expected value this way is not feasible for super large games because the branching factor is too big
        if history.is_terminal():
            return history.terminal_utility(player)
        else:
            infoSet = self.get_infoSet(history)

            average_strategy = infoSet.get_average_strategy()
            ev = 0
            for a in infoSet.actions():
                value = self.get_expected_value(history + a, (player + 1) % 2)
                ev += average_strategy[a] * value

            return ev

    def get_best_response(self, history: History, player: Player, player_strategy=None):
        """
        TODO: This only works when the action space is constant throughout. We need something more customized.

        A best response is deterministic. It is a strategy profile that chooses the action that maximizes
        its expected value.

        If player_strategy is provided, it will be computed from the player's strategy profile.

        Else, calculate the best response from the nash equilibrium.

        Cannot be a terminal history.

        returns the action index with the lowest EV (This is what the opponent should play).
        We do this by playing all actions with equal probability.


        returns (action_idx, action_ev)
        """
        assert not history.is_terminal()
        assert history.player() == player  # it should be the player's turn

        infoSet = self.get_infoSet(history)

        ev = []
        if player_strategy:
            average_strategy = player_strategy
        else:
            average_strategy = (
                infoSet.get_average_strategy()
            )  # Use the strategy computed if no strategy provided

        # Find the action that is a best response (gives the lowest EV) to player i's strategy profile

        """
		The algorithm is the following:
		1. For each action that the opponent can play after our decision, we see what happens if our opponent sticks to that strategy.
		Whatever action our opponent chooses that minimizes our expected value is the one.
		"""
        sample_a = infoSet.actions()[0]
        sample_opp_history = (
            history + sample_a
        )  # TODO: This does not work for Hold'EM? Because some actions, like folding or calling, end the current game stage, and then it's our turn again
        sample_opp_infoSet = self.get_infoSet(sample_opp_history)
        opp_actions = sample_opp_infoSet.actions()
        for opp_idx, opp_action in enumerate(opp_actions):
            # Create deterministic opponent strategy
            opp_strategy = np.zeros(len(opp_actions))
            opp_strategy[opp_idx] = 1.0
            ev_opp_action = 0
            for idx, a in enumerate(infoSet.actions()):
                value = self.get_expected_value(
                    history + a, player, average_strategy, opp_strategy=opp_strategy
                )
                ev_opp_action += average_strategy[idx] * value
                print(ev_opp_action)

            ev.append(ev_opp_action)

        br_action = np.argmin(ev)
        return br_action, min(ev)


class InfoSetTracker:
    """
    We also want to use this to track exploitability
    """

    def __init__(self):
        self.tracker_hist = []
        self.exploitability: Dict[int:float] = {}  # A dictionary of exploitability for index
        # tracker.set_histogram(f'strategy.*')
        # tracker.set_histogram(f'average_strategy.*')
        # tracker.set_histogram(f'regret.*')

    def __call__(self, infoSets: Dict[str, InfoSet]):
        self.tracker_hist.append(infoSets)

    def pprint(self):
        infoSets = self.tracker_hist[-1]
        for infoSet in infoSets.values():
            print(
                infoSet.infoSet,
                "Regret: ",
                infoSet.regret,
                "Average Strategy: ",
                infoSet.get_average_strategy(),
            )
