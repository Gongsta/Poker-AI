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
