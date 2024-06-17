import joblib
import pyttsx3
import numpy as np
from player import Player
from abstraction import calculate_equity


def getAction(strategy):
    return np.random.choice(list(strategy.keys()), p=list(strategy.values()))


class AIPlayer(Player):
    def __init__(self, balance) -> None:
        super().__init__(balance)
        self.is_AI = True

        self.engine = pyttsx3.init()

    # We are going to have the dumbest AI possible, which is to call every time
    def place_bet(self, observed_env) -> int:  # AI will call every time
        # strategy = observed_env.get_average_strategy()
        if "k" in observed_env.valid_actions():
            action = "k"
        else:
            action = "c"

        card_str = [str(card) for card in self.hand]
        community_cards = [str(card) for card in observed_env.community_cards]
        # if observed_env.game_stage == 2:
        equity = calculate_equity(card_str, community_cards)

        # fold, check / call, raise
        np_strategy = np.abs(np.array([1.0 - (equity + equity / 2.0), equity, equity / 2.0]))
        np_strategy = np_strategy / np.sum(np_strategy)  # normalize

        if observed_env.stage_pot_balance == 0:  # no bet placed
            if self == observed_env.get_player(
                observed_env.dealer_button_position
            ):  # If you are the dealer, raise more of the time
                strategy = {
                    "k": np_strategy[0],
                    f"b{min(observed_env.BIG_BLIND, self.player_balance)}": np_strategy[2],
                    f"b{min(observed_env.total_pot_balance, self.player_balance)}": np_strategy[1],
                }
            else:
                strategy = {
                    "k": equity,
                    f"b{min(observed_env.total_pot_balance, self.player_balance)}": 1 - equity,
                }

        else:  # if there is a bet already
            # TODO: calculate proportional to bet size
            # normalize the strategy
            if "k" in observed_env.valid_actions():
                strategy = {
                    "k": np_strategy[0],
                    f"b{min(int(1.5 * observed_env.get_highest_current_bet()), self.player_balance)}": np_strategy[
                        1
                    ],
                    f"b{min(2 * observed_env.get_highest_current_bet(), self.player_balance)}": np_strategy[
                        2
                    ],
                }
            else:
                strategy = {
                    "f": np_strategy[0],
                    "c": np_strategy[1],
                    f"b{min(2 * observed_env.get_highest_current_bet(), self.player_balance)}": np_strategy[
                        2
                    ],
                }

        print(card_str, community_cards)
        print(observed_env.get_highest_current_bet())
        print("equity", equity)
        print("AI strategy ", strategy)
        action = getAction(strategy)

        # history = HoldEmHistory(observed_env.history)
        # strategy = observed_env.get_average_strategy()

        # print("AI strategy", strategy)
        # print("AI action", action)

        if action == "k":  # check
            if observed_env.game_stage == 2:
                self.current_bet = 2
            else:
                self.current_bet = 0

            self.engine.say("I Check")
        elif action == "c":
            self.engine.say("I Call")
            # If you call on the preflop
            self.current_bet = observed_env.get_highest_current_bet()
        elif action == "f":
            self.engine.say("I Fold")
        else:
            self.current_bet = int(action[1:])
            if self.current_bet == self.player_balance:
                self.engine.say("I'm All-In! What are you gonna do young man?")
            self.engine.say(f"I bet {self.current_bet}")

        self.engine.runAndWait()
        return action


def load_holdem_infosets():
    print("loading holdem infosets")
    global holdem_infosets
    # holdem_infosets = joblib.load("../src/infoSets_100.joblib")
    holdem_infosets = joblib.load("../src/infoSets_0.joblib")
    print("loaded holdem infosets!")


def get_infoset(infoSet_key):
    print("getting infoset", infoSet_key)
    key = "".join(infoSet_key)
    if key in holdem_infosets:
        return holdem_infosets[key]
    else:
        return None
