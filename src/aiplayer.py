import random
import pyttsx3
import numpy as np
from player import Player
from abstraction import calculate_equity
from preflop_holdem import PreflopHoldemHistory, PreflopHoldemInfoSet
from postflop_holdem import PostflopHoldemHistory, PostflopHoldemInfoSet
import joblib
import copy


def getAction(strategy):
    return np.random.choice(list(strategy.keys()), p=list(strategy.values()))


class AIPlayer(Player):
    def __init__(self, balance) -> None:
        super().__init__(balance)
        self.is_AI = True
        self.speak = True

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
                "I Fold. You win this one, but not for long.",
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
        if self.speak:
            self.engine.runAndWait()

    def trash_talk_lose(self):
        self.engine.say(random.choice(self.get_trash_talk("lose")))
        if self.speak:
            self.engine.runAndWait()

    def trash_talk_fold(self):
        self.engine.say(random.choice(self.get_trash_talk("opponent_fold")))
        if self.speak:
            self.engine.runAndWait()

    def process_action(self, action, observed_env):
        if action == "k":  # check
            if observed_env.game_stage == 2:
                self.current_bet = 2
            else:
                self.current_bet = 0

            self.engine.say(random.choice(self.get_trash_talk("k")))
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

        if self.speak:
            self.engine.runAndWait()

    def place_bet(self, observed_env):
        raise NotImplementedError


class EquityAIPlayer(AIPlayer):
    def __init__(self, balance) -> None:
        super().__init__(balance)

    def place_bet(self, observed_env) -> int:  # AI will call every time
        """
        A Strategy implemented with human heuristics.
        """
        if "k" in observed_env.valid_actions():
            action = "k"
        else:
            action = "c"

        card_str = [str(card) for card in self.hand]
        community_cards = [str(card) for card in observed_env.community_cards]

        isDealer = self == observed_env.get_player(observed_env.dealer_button_position)
        checkAllowed = "k" in observed_env.valid_actions()

        action = self.get_action(
            card_str,
            community_cards,
            observed_env.total_pot_balance,
            observed_env.get_highest_current_bet(),
            observed_env.BIG_BLIND,
            self.player_balance,
            isDealer,
            checkAllowed,
        )

        self.process_action(action, observed_env)  # use voice activation
        return action

    def get_action(
        self,
        card_str,
        community_cards,
        total_pot_balance,
        highest_current_bet,
        BIG_BLIND,
        player_balance,
        isDealer,
        checkAllowed,
    ):
        equity = calculate_equity(card_str, community_cards)
        # fold, check / call, raise
        np_strategy = np.abs(np.array([1.0 - (equity + equity / 2.0), equity, equity / 2.0]))
        np_strategy = np_strategy / np.sum(np_strategy)  # normalize

        if highest_current_bet == 0:  # no bet placed
            if isDealer:  # If you are the dealer, raise more of the time
                strategy = {
                    "k": np_strategy[0],
                    f"b{min(max(BIG_BLIND, int(total_pot_balance / 3)), player_balance)}": np_strategy[
                        2
                    ],
                    f"b{min(total_pot_balance, player_balance)}": np_strategy[1],
                }
            else:
                strategy = {
                    "k": equity,
                    f"b{min(total_pot_balance, player_balance)}": 1 - equity,
                }

        else:  # if there is a bet already
            if checkAllowed:
                strategy = {
                    "k": np_strategy[0],
                    f"b{min(int(1.5 * highest_current_bet), player_balance)}": np_strategy[1],
                    f"b{min(2 * highest_current_bet, player_balance)}": np_strategy[2],
                }
            else:
                if highest_current_bet == player_balance:
                    strategy = {
                        "f": np_strategy[0],
                        "c": np_strategy[1] + np_strategy[2],
                    }
                else:
                    strategy = {
                        "f": np_strategy[0],
                        "c": np_strategy[1],
                        f"b{min(2 * highest_current_bet, player_balance)}": np_strategy[2],
                    }

        # renormalize the strategy in case of duplicates
        total = sum(strategy.values())
        for key in strategy:
            strategy[key] /= total
        action = getAction(strategy)
        return action


class CFRAIPlayer(AIPlayer):
    def __init__(self, balance) -> None:
        super().__init__(balance)

        self.preflop_infosets = joblib.load("../src/preflop_infoSets_batch_19.joblib")
        self.postflop_infosets = joblib.load("../src/postflop_infoSets_batch_19.joblib")

    def place_bet(self, observed_env):
        card_str = [str(card) for card in self.hand]
        community_cards = [str(card) for card in observed_env.community_cards]

        isDealer = self == observed_env.get_player(observed_env.dealer_button_position)
        checkAllowed = "k" in observed_env.valid_actions()

        action = self.get_action(
            observed_env.history,
            card_str,
            community_cards,
            observed_env.get_highest_current_bet(),
            observed_env.stage_pot_balance,
            observed_env.total_pot_balance,
            self.player_balance,
            observed_env.BIG_BLIND,
            isDealer,
            checkAllowed,
        )
        self.process_action(action, observed_env)  # use voice activation
        return action

    def get_action(
        self,
        history,
        card_str,
        community_cards,
        highest_current_bet,
        stage_pot_balance,
        total_pot_balance,
        player_balance,
        BIG_BLIND,
        isDealer,
        checkAllowed,
    ):

        # Bet sizing uses the pot balance
        # stage_pot_balance used for preflop, total_pot_balance used for postflop

        action = None
        HEURISTICS = True  # trying this in case my preflop strategy sucks
        if len(community_cards) == 0:  # preflop
            if HEURISTICS:
                player = EquityAIPlayer(self.player_balance)
                action = player.get_action(
                    card_str,
                    community_cards,
                    total_pot_balance,
                    highest_current_bet,
                    BIG_BLIND,
                    player_balance,
                    isDealer,
                    checkAllowed,
                )
            else:
                abstracted_history = self.perform_preflop_abstraction(history, BIG_BLIND=BIG_BLIND)
                infoset_key = "".join(PreflopHoldemHistory(abstracted_history).get_infoSet_key())
                strategy = self.preflop_infosets[infoset_key].get_average_strategy()
                abstracted_action = getAction(strategy)
                if abstracted_action == "bMIN":
                    action = "b" + str(max(BIG_BLIND, int(stage_pot_balance)))
                elif abstracted_action == "bMID":
                    action = "b" + str(max(BIG_BLIND, 2 * int(stage_pot_balance)))
                elif (
                    abstracted_action == "bMAX"
                ):  # in training, i have it set to all in... but wiser to 4x pot?
                    action = "b" + str(min(player_balance, 4 * int(stage_pot_balance)))
                else:
                    action = abstracted_action
        else:
            abstracted_history = self.perform_postflop_abstraction(
                history, BIG_BLIND=BIG_BLIND
            )  # condense down bet sequencing
            infoset_key = PostflopHoldemHistory(abstracted_history).get_infoSet_key_online()
            strategy = self.postflop_infosets[infoset_key].get_average_strategy()
            abstracted_action = getAction(strategy)
            print("Abstracted action: ", action)
            if abstracted_action == "bMIN":
                action = "b" + str(max(BIG_BLIND, int(1 / 3 * total_pot_balance)))
            elif abstracted_action == "bMAX":
                action = "b" + str(min(total_pot_balance, player_balance))
            else:
                action = abstracted_action

        print("history: ", history)
        if not HEURISTICS:
            print("Abstracted history: ", abstracted_history)
            print("Infoset key: ", infoset_key)
            print("AI strategy ", strategy)
            print("Abstracted Action:", abstracted_action, "Final Action:", action)

        return action

    def perform_preflop_abstraction(self, history, BIG_BLIND=2):
        stage = copy.deepcopy(history)
        abstracted_history = stage[:2]
        if (
            len(stage) >= 6 and stage[3] != "c"  # bet seqeuence of length 4
        ):  # length 6 that isn't a call, we need to condense down
            if len(stage) % 2 == 0:
                abstracted_history += ["bMAX"]
            else:
                abstracted_history += ["bMIN", "bMAX"]
        else:
            bet_size = BIG_BLIND
            pot_total = 3
            for i, action in enumerate(stage[2:]):
                if action[0] == "b":
                    bet_size = int(action[1:])

                    # this is a raise on a small bet
                    if abstracted_history[-1] == "bMIN":
                        if bet_size <= 2 * pot_total:
                            abstracted_history += ["bMID"]
                        else:
                            abstracted_history += ["bMAX"]
                    elif abstracted_history[-1] == "bMID":
                        abstracted_history += ["bMAX"]
                    elif abstracted_history[-1] == "bMAX":
                        if abstracted_history[-2] == "bMID":
                            abstracted_history[-2] = "bMIN"
                        abstracted_history[-1] = "bMID"
                        abstracted_history += ["bMAX"]
                    else:  # first bet
                        if bet_size <= pot_total:
                            abstracted_history += ["bMIN"]
                        elif bet_size <= 2 * pot_total:
                            abstracted_history += ["bMID"]
                        else:
                            abstracted_history += ["bMAX"]

                    pot_total += bet_size

                elif action == "c":
                    pot_total = 2 * bet_size
                    abstracted_history += ["c"]
                else:
                    abstracted_history += [action]
        return abstracted_history

    def perform_postflop_abstraction(self, history, BIG_BLIND=2):
        history = copy.deepcopy(history)

        pot_total = BIG_BLIND * 2
        # Compute preflop pot size
        flop_start = history.index("/")
        for i, action in enumerate(history[:flop_start]):
            if action[0] == "b":
                bet_size = int(action[1:])
                pot_total = 2 * bet_size

        # ------- Remove preflop actions + bet abstraction -------
        abstracted_history = history[:2]
        # swap dealer and small blind positions for abstraction
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
                    if len(stage) % 2 == 0:
                        abstracted_history += ["bMAX"]
                    else:
                        abstracted_history += ["bMIN", "bMAX"]
            else:
                for i, action in enumerate(stage):
                    if action[0] == "b":
                        bet_size = int(action[1:])
                        latest_bet = bet_size

                        # this is a raise on a small bet
                        if abstracted_history[-1] == "bMIN":
                            abstracted_history += ["bMAX"]
                        # this is a raise on a big bet
                        elif (
                            abstracted_history[-1] == "bMAX"
                        ):  # opponent raised, first bet must be bMIN
                            abstracted_history[-1] = "bMIN"
                            abstracted_history += ["bMAX"]
                        else:  # first bet
                            if bet_size >= pot_total:
                                abstracted_history += ["bMAX"]
                            else:
                                abstracted_history += ["bMIN"]

                        pot_total += bet_size

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
