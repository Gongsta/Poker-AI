import random
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

    def get_trash_talk(self, action_type, bet_amount=0):
        trash_talk = {
            "k": [
                "I Check. Don't get too excited now.",
                "Check. Let's see what you've got.",
                "I'll check. Not impressed yet.",
            ],
            "c": [
                "I Call. You think you're tough?",
                "Call. Let's see what you're hiding.",
                "I call your bet. Bring it on!",
            ],
            "f": [
                "I Fold. You win this one.",
                "Fold. Consider yourself lucky.",
                "I'm folding. Don't get used to it.",
            ],
            "all_in": [
                "I'm All-In! What are you gonna do, young man?",
                "All-In! Show me what you got!",
                "I'm going All-In! Can you handle it?",
            ],
            "b": [
                f"I bet {bet_amount}$. Do you have the guts?",
                f"Bet {bet_amount}$. Let's up the stakes!",
                f"I'm betting {bet_amount}$. Feeling lucky?",
            ],
            "win": [
                "I win! Better luck next time.",
                "Victory is sweet. Did you even stand a chance?",
                "I told you, I'm the best. Pay up!",
            ],
            "lose": [
                "You win this time. Don't get used to it.",
                "Lucky break. Enjoy it while it lasts.",
                "You got me this time. It won't happen again.",
            ],
            "opponent_fold": [
                "Hah! Folding already? Pathetic.",
                "You're folding? I expected more fight from you.",
                "Fold? I knew you couldn't handle the pressure.",
            ],
        }
        return trash_talk[action_type]

    def trash_talk_win(self):
        self.engine.say(random.choice(self.get_trash_talk("win")))
        self.engine.runAndWait()

    def trash_talk_lose(self):
        self.engine.say(random.choice(self.get_trash_talk("lose")))
        self.engine.runAndWait()

    def trash_talk_fold(self):
        self.engine.say(random.choice(self.get_trash_talk("opponent_fold")))
        self.engine.runAndWait()

    def process_action(self, action, observed_env):
        if action == "k":  # check
            if observed_env.game_stage == 2:
                self.current_bet = 2
            else:
                self.current_bet = 0

            self.engine.say("I Check")
        elif action == "c":
            if observed_env.get_highest_current_bet() == self.player_balance:
                self.engine.say("I call your all-in. You think I'm afraid?")
            else:
                self.engine.say(random.choice(self.get_trash_talk("c")))
            # If you call on the preflop
            self.current_bet = observed_env.get_highest_current_bet()
        elif action == "f":
            self.engine.say(random.choice(self.get_trash_talk("f")))
        else:
            self.current_bet = int(action[1:])
            if self.current_bet == self.player_balance:
                self.engine.say(random.choice(self.get_trash_talk("all_in")))
            else:
                self.engine.say(random.choice(self.get_trash_talk("b", self.current_bet)))

        self.engine.runAndWait()

    def place_bet(self, observed_env):
        raise NotImplementedError


class EquityAIPlayer(AIPlayer):
    def __init__(self, balance) -> None:
        super().__init__(balance)

    def place_bet(self, observed_env) -> int:  # AI will call every time
        """
        A Strategy implemented with human heuristics
        """
        if "k" in observed_env.valid_actions():
            action = "k"
        else:
            action = "c"

        card_str = [str(card) for card in self.hand]
        community_cards = [str(card) for card in observed_env.community_cards]

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
                    f"b{min(max(observed_env.BIG_BLIND, int(observed_env.total_pot_balance / 3)), self.player_balance)}": np_strategy[
                        2
                    ],
                    f"b{min(observed_env.total_pot_balance, self.player_balance)}": np_strategy[1],
                }
            else:
                strategy = {
                    "k": equity,
                    f"b{min(observed_env.total_pot_balance, self.player_balance)}": 1 - equity,
                }

        else:  # if there is a bet already
            # TODO: calculate proportional to bet size. i mean, it just does it according to bet size
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
                if observed_env.get_highest_current_bet() == self.player_balance:
                    strategy = {
                        "f": np_strategy[0],
                        "c": np_strategy[1] + np_strategy[2],
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
        self.process_action(action, observed_env)
        return action


import joblib
from abstraction import calculate_equity, predict_cluster_fast
from postflop_holdem import HoldemInfoSet, HoldEmHistory

import copy


class CFRAIPlayer(AIPlayer):
    def __init__(self, balance) -> None:
        super().__init__(balance)

        self.infosets = joblib.load("../src/infoSets_batch_7.joblib")

    def perform_postflop_abstraction(self, observed_env):
        history = copy.deepcopy(observed_env.history)

        pot_total = observed_env.BIG_BLIND * 2
        # Compute preflop pot size
        flop_start = history.index("/")
        for i, action in enumerate(history[:flop_start]):
            if action[0] == "b":
                bet_size = int(action[1:])
                pot_total = 2 * bet_size

        # Remove preflop actions
        abstracted_history = history[:2]

        # Bet Abstraction (card abstraction is done later)
        stage_start = flop_start
        stage = self.get_stage(history[stage_start + 1 :])
        latest_bet = 0
        while True:
            abstracted_history += ["/"]

            if (
                len(stage) >= 4 and stage[3] != "c"
            ):  # length 4 that isn't a call, we need to condense down
                abstracted_history += [stage[0]]

                if stage[-1] == "c":
                    if len(stage) % 2 == 1:  # ended on dealer
                        abstracted_history += ["bMAX", "c"]
                    else:
                        if stage[0] == "k":
                            abstracted_history += ["k", "bMAX", "c"]
                        else:
                            abstracted_history += ["bMIN", "bMAX", "c"]
            else:
                for i, action in enumerate(stage):
                    if action[0] == "b":
                        bet_size = int(action[1:])
                        latest_bet = bet_size
                        pot_total += bet_size

                        # this is a raise on a small bet
                        if abstracted_history[-1] == "bMIN":
                            abstracted_history += ["bMAX"]
                        # this is a raise on a big bet
                        elif abstracted_history[-1] == "bMAX":
                            abstracted_history[-1] = "k"  # turn into a check
                        else:  # first bet
                            if bet_size >= pot_total:
                                abstracted_history += ["bMAX"]
                            else:
                                abstracted_history += ["bMIN"]

                    elif action == "c":
                        pot_total += latest_bet
                        abstracted_history += ["c"]
                    else:
                        abstracted_history += [action]

            # Proceed to next stage or exit if final stage
            if "/" not in history[stage_start + 1 :]:
                break
            stage_start = history[stage_start + 1 :].index("/") + (stage_start + 1)
            stage = self.get_stage(history[stage_start + 1 :])

        return abstracted_history

    def get_stage(self, history):
        if "/" in history:
            return history[: history.index("/")]
        else:
            return history

    def place_bet(self, observed_env):
        if observed_env.game_stage == 2:  # preflop
            if "k" in observed_env.valid_actions():
                action = "k"
            else:
                action = "c"
        else:
            abstracted_history = self.perform_postflop_abstraction(observed_env)
            print("abstracted history", abstracted_history)
            infoset_key = HoldEmHistory(abstracted_history).get_infoSet_key_online()
            strategy = self.infosets[infoset_key].get_average_strategy()
            print(infoset_key)
            print("AI strategy ", strategy)
            action = getAction(strategy)
            if action == "bMIN":
                action = "b" + str(
                    max(observed_env.BIG_BLIND, int(1 / 3 * observed_env.total_pot_balance))
                )
            elif action == "bMAX":
                action = "b" + str(min(observed_env.total_pot_balance, self.player_balance))

            self.process_action(action, observed_env)
        return action


# def load_holdem_infosets():
#     print("loading holdem infosets")
#     global holdem_infosets
#     # holdem_infosets = joblib.load("../src/infoSets_100.joblib")
#     holdem_infosets = joblib.load("../src/infoSets_0.joblib")
#     print("loaded holdem infosets!")


# def get_infoset(infoSet_key):
#     print("getting infoset", infoSet_key)
#     key = "".join(infoSet_key)
#     if key in holdem_infosets:
#         return holdem_infosets[key]
#     else:
#         return None
