import joblib
import pyttsx3
import numpy as np
from evaluator import *
from typing import List


class Player:  # This is the POV
    def __init__(self, balance) -> None:
        self.is_AI = False

        self.hand: List[Card] = (
            []
        )  # The hand is also known as hole cards: https://en.wikipedia.org/wiki/Texas_hold_%27em
        self.player_balance: int = (
            balance  # TODO: Important that this value cannot be modified easily...
        )
        self.current_bet = 0
        self.playing_current_round = True

    # Wellformedness, hand is always either 0 or 2 cards
    def add_card_to_hand(self, card: Card):
        self.hand.append(card)
        assert len(self.hand) <= 2

    def clear_hand(self):
        self.hand = []

    def place_bet(self, action: str, observed_env) -> int:
        if action == "c":
            self.current_bet = observed_env.get_highest_current_bet()

        elif action[0] == "b":  # bet X amount
            bet_size = int(action[1:])
            if bet_size < observed_env.get_highest_current_bet():
                print("you must raise more than the current highest bet")
                return None
            elif bet_size > self.player_balance:
                print("you cannot bet more than your balance")
                return None
            elif bet_size == observed_env.get_highest_current_bet():
                print("you must call, not bet")
            else:
                self.current_bet = int(action[1:])

        return action


def getAction(strategy):
    return np.random.choice(list(strategy.keys()), p=list(strategy.values()))


class AIPlayer(Player):
    def __init__(self, balance) -> None:
        super().__init__(balance)
        self.is_AI = True

        self.engine = pyttsx3.init()

    # We are going to have the dumbest AI possible, which is to call every time
    def place_bet(self, observed_env) -> int:  # AI will call every time
        print(observed_env)
        # Very similar function to Player.place_bet, we only call and check
        # use the the history
        # strategy = observed_env.get_average_strategy()
        if "k" in observed_env.valid_actions():
            action = "k"
        else:
            action = "c"

        print(observed_env.history)

        # action = getAction(strategy)
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
            self.engine.say(f"I bet {self.current_bet * 100}")

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
