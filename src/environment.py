# The Poker Environment
from evaluator import *
from typing import List
from holdem import HoldEmHistory, HoldemInfoSet  # To get the legal actions


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
        hist: HoldEmHistory = observed_env.history
        legal_actions = hist.actions()
        if action not in legal_actions:
            raise Exception(f"Invalid Action: {action}")

        current_game_stage_history, stage = hist.get_current_game_stage_history()

        if action == "k":  # check
            if stage == "preflop":
                self.current_bet = 2  # BB
            else:
                self.current_bet = 0

        elif action == "c":
            # If you call on the preflop
            if len(hist.history) == 2:
                self.current_bet = 2
            else:
                self.current_bet = int(hist.history[-1][1:])

        elif action[0] == "b":  # bet X amount
            self.current_bet = int(action[1:])
        return action

    def calculate_pot_odds(
        self,
    ):  # Calculate Pot Odds helper function, basically look at how many hands can you currently beat
        """
        Simple logic, does not account for the pot values.
        """


class AIPlayer(Player):
    def __init__(self, balance) -> None:
        super().__init__(balance)
        self.is_AI = True

    # We are going to have the dumbest AI possible, which is to call every time
    def place_bet(self, observed_env) -> int:  # AI will call every time
        # Very similar function to Player.place_bet, we only call and check
        action = "k"
        hist: HoldEmHistory = observed_env.history
        legal_actions = hist.actions()
        if action not in legal_actions:
            action = "c"

        if action not in legal_actions:
            raise Exception("AI found no legal actions", hist.actions())

        current_game_stage_history, stage = hist.get_current_game_stage_history()
        if action == "k":  # check
            if stage == "preflop":
                self.current_bet = 2
            else:
                self.current_bet = 0

        elif action == "c":
            # If you call on the preflop
            if len(hist.history) == 2:
                self.current_bet = observed_env.big_blind
            else:  # Set the current bet to the amount of the last bet
                self.current_bet = int(hist.history[-1][1:])

        return action


class PokerEnvironment:
    """
    Also see the HoldEmHistory class in holdem.py, which defines the set of legal actions every time
    """

    def __init__(self) -> None:
        self.players: List[Player] = []
        self.deck = Deck()

        """Game Stages:
		1: Starting a new round, giving players their cards. Automatically goes into state 2
		2: Preflop betting round. Goes into state 3 once everyone has made their decision
		3: Flop round. Goes into turn (state 4) /ends round (state 6) once everyone " " 
		4: Turn round. Goes into river (state 5) /ends round (state 6) once everyone " " 
		5: River round. Ends round (state 6) once everyone " " 
		6: Round is over. Distribute pot winnings.
		
		Game Stage - 2 = number of "/" in the holdem infoset and history
		"""
        self.game_stage = 1  # To keep track of which phase of the game we are at, new_round is 0
        # If self.finished_playing_game_stage = True, we can move to the next game state. This is needed to go around each player and await their decision
        self.finished_playing_game_stage = False

        # Changes every round
        self.dealer_button_position = 0  # This button will move every round
        self.total_pot_balance = 0  # keep track of pot size of total round
        self.stage_pot_balance = 0  # keep track of pot size for current round
        self.community_cards: List[Card] = []  # a.k.a. the board
        self.position_in_play = 0

        self.first_player_to_place_highest_bet = 0  # This is to keep track of who is the first player to have placed the highest bet, so we know when to end the round

        # These values should rarely change. TODO: Figure out how to integrate with holdem.py
        self.new_player_balance = 100
        self.small_blind = 1
        self.big_blind = 2
        self.history: HoldEmHistory = (
            HoldEmHistory()
        )  # THis will be the history that will be fed into the AI

        self.players_balance_history = []  # List of "n" list for "n" players

    def add_player(self):
        self.players.append(Player(self.new_player_balance))

    def get_player(self, idx) -> Player:
        return self.players[idx]

    def add_AI_player(self):  # Add a dumb AI
        self.players.append(AIPlayer(self.new_player_balance))

    def get_winning_players(self) -> List:
        # If there is more than one winning player, the pot is split. We assume that we only run things once
        winning_players: List = []
        for player in self.players:
            if player.playing_current_round:
                winning_players.append(player)

        return winning_players

    def get_winning_players_idx(self) -> List:
        # If there is more than one winning player, the pot is split. We assume that we only run things once
        winning_players: List = []
        for idx, player in enumerate(self.players):
            if player.playing_current_round:
                winning_players.append(idx)

        return winning_players

    def distribute_pot_to_winning_players(self):  # Run when self.game_stage = 5
        winning_players = self.get_winning_players()

        pot_winning = self.total_pot_balance / len(winning_players)
        for player in winning_players:
            player.player_balance += pot_winning

        # Used for graphing later
        for idx, player in enumerate(self.players):
            # TODO: To be changed if you want to keep the balance history until the very end
            try:
                self.players_balance_history[idx].append(
                    int(player.player_balance - self.new_player_balance)
                )
            except:
                self.players_balance_history.append([])
                self.players_balance_history[idx].append(
                    int(player.player_balance - self.new_player_balance)
                )

        self.total_pot_balance = 0  # Reset the pot just to be safe
        self.stage_pot_balance = 0  # Reset the pot just to be safe

    def count_remaining_players_in_round(self):
        # Helper function to count the total number of players still in the round
        total = 0
        for player in self.players:
            if player.playing_current_round:
                total += 1
        return total

    def print_board(self):
        for card in self.community_cards:
            card.print()

    def start_new_round(self):
        self.showdown = False
        assert len(self.players) >= 2  # We cannot start a poker round with less than 2 players...

        # Reset Players
        for player in self.players:
            player.playing_current_round = True
            player.current_bet = 0
            player.clear_hand()
            # TODO: Remove this when you are ready
            player.player_balance = self.new_player_balance

        # Reset Deck (shuffles it as well), reset pot size
        self.deck.reset_deck()
        self.community_cards = []
        self.stage_pot_balance = 0
        self.total_pot_balance = 0

        self.history = HoldEmHistory()  # Reset the history

        # Move the dealer position and assign the new small and big blinds
        self.dealer_button_position += 1
        self.dealer_button_position %= len(self.players)

        # Big Blind
        self.players[((self.dealer_button_position + 1) % len(self.players))].current_bet = (
            self.big_blind
        )

        # Small Blind
        self.players[((self.dealer_button_position + 2) % len(self.players))].current_bet = (
            self.small_blind
        )

        self.update_stage_pot_balance()
        # 3. Deal Cards
        # We start dealing with the player directly clockwise of the dealer button
        position_to_deal = self.dealer_button_position + 1

        for _ in range(len(self.players)):
            position_to_deal %= len(self.players)
            card_str = ""
            for _ in range(2):
                card = self.deck.draw()
                card_str += str(card)
                self.players[position_to_deal].add_card_to_hand(card)

            self.history += card_str
            position_to_deal += 1

        self.finished_playing_game_stage = True

    def update_stage_pot_balance(self):
        """
        Assumes the balances from the players are correct

        """
        self.stage_pot_balance = 0
        for player in self.players:
            self.stage_pot_balance += player.current_bet

    def play_current_stage(self, action: str = ""):
        self.update_stage_pot_balance()
        if self.players[self.position_in_play].is_AI:
            action = self.players[self.position_in_play].place_bet(
                self
            )  # Pass the Environment as an argument
            self.history += action

        else:  # Real player's turn
            if action == "":  # No decision has yet been made
                return
            else:
                self.players[self.position_in_play].place_bet(action, self)
                # Update the history
                self.history += action

        if action[0] == "b":
            self.first_player_to_place_highest_bet = self.position_in_play

        elif action == "f":
            self.players[self.position_in_play].playing_current_round = False  # Player has folded
        self.update_stage_pot_balance()

        if self.count_remaining_players_in_round() == 1:  # Round is over, distribute winnings
            self.finished_playing_game_stage = True
            self.game_stage = 6
            return
        else:
            self.move_to_next_player()

        if (
            self.position_in_play == self.first_player_to_place_highest_bet
        ):  # Stage is over, move to the next stage (see flop)
            self.finished_playing_game_stage = True

    def move_to_next_player(self):
        assert self.count_remaining_players_in_round() > 1
        self.position_in_play += 1
        self.position_in_play %= len(self.players)

        while not self.players[self.position_in_play].playing_current_round:
            self.position_in_play += 1
            self.position_in_play %= len(self.players)

    def play_preflop(self):
        """
        About the small blind position:
        The "small blind" is placed by the player to the left of the dealer button and the "big blind" is then posted by the next player to the left.
        The one exception is when there are only two players (a "heads-up" game), when the player on the button is the small blind, and the other player is the big blind.
        """
        if len(self.players) == 2:
            self.position_in_play = self.dealer_button_position
        else:
            self.position_in_play = (self.dealer_button_position + 3) % len(self.players)
        self.first_player_to_place_highest_bet = self.position_in_play

        self.finished_playing_game_stage = False

    def play_flop(self):
        # 3. Flop
        self.history += "/"

        self.deck.draw()  # We must first burn one card, TODO: Show on video

        cards = ""
        for _ in range(3):  # Draw 3 cards
            self.community_cards.append(self.deck.draw())
            cards += str(self.community_cards[-1])

        self.history += cards

        # The person that should play is the first person after the dealer position
        self.position_in_play = self.dealer_button_position
        self.move_to_next_player()
        self.first_player_to_place_highest_bet = self.position_in_play

        self.finished_playing_game_stage = False

    def play_turn(self):
        # 4. Turn
        self.history += "/"

        self.deck.draw()  # We must first burn one card, TODO: Show on video
        self.community_cards.append(self.deck.draw())

        self.history += str(self.community_cards[-1])

        # The person that should play is the first person after the dealer position
        self.position_in_play = self.dealer_button_position
        self.move_to_next_player()
        self.first_player_to_place_highest_bet = self.position_in_play

        self.finished_playing_game_stage = False

    def play_river(self):
        # 5. River
        self.history += "/"

        self.deck.draw()  # We must first burn one card, TODO: Show on video
        self.community_cards.append(self.deck.draw())

        self.history += str(self.community_cards[-1])

        self.finished_playing_game_stage = False

    def update_player_balances_at_end_of_stage(self):
        for player in self.players:
            player.player_balance -= player.current_bet
            player.current_bet = 0

    def move_stage_to_total_pot_balance(self):
        self.total_pot_balance += self.stage_pot_balance
        self.stage_pot_balance = 0

    def handle_game_stage(self, action=""):
        if self.finished_playing_game_stage:
            if self.game_stage != 1:
                self.update_player_balances_at_end_of_stage()
                self.move_stage_to_total_pot_balance()
            self.game_stage += 1

            if self.game_stage == 2:
                self.play_preflop()
            elif self.game_stage == 3:
                self.play_flop()
            elif self.game_stage == 4:
                self.play_turn()
            elif self.game_stage == 5:
                self.play_river()
            else:
                if (
                    self.game_stage == 6
                ):  # We reached the river, and are now in the showdown. We need the evaluator to get the winners, set all losers to playing_current_round false
                    self.showdown = True
                    evaluator = Evaluator()

                    indices_of_potential_winners = []
                    for idx, player in enumerate(self.players):
                        if player.playing_current_round:
                            indices_of_potential_winners.append(idx)
                            hand = CombinedHand(self.community_cards + player.hand)
                            evaluator.add_hands(hand)

                    winners = evaluator.get_winner()
                    for player in self.players:
                        player.playing_current_round = False

                    for winner in winners:
                        self.players[indices_of_potential_winners[winner]].playing_current_round = (
                            True
                        )

                self.game_stage = 1
                self.finished_playing_game_stage = (
                    False  # on the next call of the handler, we will start a new round
                )
        else:
            if self.game_stage == 1:
                # This function was put here instead of at game_stage == 6 to visualize the game
                self.distribute_pot_to_winning_players()
                self.start_new_round()
            else:
                self.play_current_stage(action)

    def end_of_round(self):
        return self.game_stage == 1 and self.finished_playing_game_stage == False
