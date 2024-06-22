# The Poker Environment
from evaluator import *
from typing import List
from player import Player
from preflop_holdem import PreflopHoldemHistory, PreflopHoldemInfoSet
from postflop_holdem import PostflopHoldemHistory, PostflopHoldemInfoSet
from aiplayer import CFRAIPlayer


class PokerEnvironment:
    """
    Also see the HoldEmHistory class in holdem.py
    """

    def __init__(self) -> None:
        self.players: List[Player] = []
        self.deck = Deck()

        """Game Stages (move_to_next_game_stage() is called to transition between stages (except for stage 1)):
		1: Initial stage. Call start_new_round() to enter the preflop stage
		2: Preflop betting round. Goes into state 3 once everyone has made their decision
		3: Flop round. Goes into turn (state 4) /ends round (state 6) once everyone " "
		4: Turn round. Goes into river (state 5) /ends round (state 6) once everyone " "
		5: River round. Ends round (state 6) once everyone " "
		6: Round is over. Distribute pot winnings. Call start_new_round() to start a new round

		Game Stage - 2 = number of "/" in the holdem infoset and history
		"""

        self.game_stage = 1  # To keep track of which phase of the game we are at, new_round is 0
        # Changes every round
        self.dealer_button_position = 0  # This button will move every round
        self.position_in_play = 0

        self.total_pot_balance = 0  # keep track of pot size of total round
        self.stage_pot_balance = 0  # keep track of pot size for current round
        self.community_cards: List[Card] = []  # a.k.a. the board

        self.raise_position = 0  # This is to keep track of who is the first player to have placed the highest bet, so we know when to end the round
        self.showdown = False  # flag that can be used to reveal opponents cards if needed

        # FIXED BALANCES
        self.new_player_balance = 2500
        self.SMALL_BLIND = 10
        self.BIG_BLIND = 20

        self.INPUT_CARDS = True

        self.history = []
        self.players_balance_history = []  # List of "n" list for "n" players

    def add_player(self):
        self.players.append(Player(self.new_player_balance))

    def get_player(self, idx) -> Player:
        return self.players[idx]

    def add_AI_player(self):  # Add a dumb AI
        self.players.append(CFRAIPlayer(self.new_player_balance))
        self.AI_player_idx = len(self.players) - 1

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
        assert self.game_stage == 6
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

    def valid_actions(self):
        """
        Mostly just enables checking whether it is allowed to check, or call.
        """
        valid_actions = ["f"]
        if self.players[0].current_bet == self.players[1].current_bet:
            valid_actions.append("k")
        else:
            valid_actions.append("c")

        return valid_actions

    def count_remaining_players_in_round(self):
        # Helper function to count the total number of players still in the round
        total = 0
        for player in self.players:
            if player.playing_current_round:
                total += 1
        return total

    def start_new_round(self):
        assert len(self.players) >= 2  # We cannot start a poker round with less than 2 players...

        if self.INPUT_CARDS:
            self.new_player_balance = int(input("Enter the starting balance for the players: "))
        # Reset Players
        for player in self.players:
            player.playing_current_round = True
            player.current_bet = 0
            player.clear_hand()
            player.player_balance = self.new_player_balance

        # Reset Deck (shuffles it as well), reset pot size
        self.deck.reset_deck()
        self.community_cards = []
        self.stage_pot_balance = 0
        self.total_pot_balance = 0

        # Move the dealer position and assign the new small and big blinds
        self.dealer_button_position += 1
        self.dealer_button_position %= len(self.players)

        self.showdown = False
        self.history = []  # reset history for the round

        # Proceed to preflop
        self.game_stage = 1
        self.move_to_next_game_stage()

    def get_highest_current_bet(self):
        highest_bet = 0
        for player in self.players:
            if player.current_bet > highest_bet and player.playing_current_round:
                highest_bet = player.current_bet

        return highest_bet

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
            action = self.players[self.position_in_play].place_bet(self)

        else:  # Real player's turn
            if action == "":  # No decision has yet been made
                return
            else:
                action = self.players[self.position_in_play].place_bet(action, self)
                if action is None:  # invalid action
                    return

        self.history += [action]

        if action[0] == "b":
            self.raise_position = self.position_in_play
        elif action == "f":
            self.players[self.position_in_play].playing_current_round = False  # Player has folded
        self.update_stage_pot_balance()

        # ---- Terminate Round if 1 player left ------
        if self.count_remaining_players_in_round() == 1:
            self.end_round()
            return

        self.move_to_next_playing_player()

        if self.position_in_play == self.raise_position:  # Everyone has called with no new raises
            self.move_to_next_game_stage()

    def move_to_next_playing_player(self, from_position=None):
        assert self.count_remaining_players_in_round() > 1
        if from_position is not None:
            self.position_in_play = from_position
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
        # Set the blind values
        # Big Blind
        self.players[((self.dealer_button_position + 1) % len(self.players))].current_bet = (
            self.BIG_BLIND
        )
        # Small Blind
        self.players[((self.dealer_button_position + 2) % len(self.players))].current_bet = (
            self.SMALL_BLIND
        )

        self.update_stage_pot_balance()

        if len(self.players) == 2:
            self.position_in_play = self.dealer_button_position
        else:
            self.position_in_play = (self.dealer_button_position + 3) % len(self.players)
        self.raise_position = self.position_in_play

        for i in range(len(self.players)):
            # First card is non-dealer, second is dealer
            player_idx = (self.dealer_button_position + 1 + i) % len(self.players)
            card_str = ""
            for i in range(2):
                if self.INPUT_CARDS and player_idx == 0:
                    card = Card(input(f"Enter the card that was dealt (ex: Ah): "))
                else:
                    card = self.deck.draw()

                card_str += str(card)
                self.players[player_idx].add_card_to_hand(card)

            self.history += [card_str]  # always deal to the non-dealer first

    def play_flop(self):
        self.deck.draw()  # We must first burn one card, TODO: Show on video

        for i in range(3):  # Draw 3 cards
            if self.INPUT_CARDS:
                card = Card(input(f"Input the {i}-th community card (ex: 'Ah'): "))
            else:
                card = self.deck.draw()

            self.community_cards.append(card)

        self.history += ["/"]
        self.history += ["".join([str(card) for card in self.community_cards])]

        # The person that should play is the first person after the dealer position
        self.move_to_next_playing_player(from_position=self.dealer_button_position)
        self.raise_position = self.position_in_play

    def play_turn(self):

        self.deck.draw()
        if self.INPUT_CARDS:
            card = Card(input("Input the turn card (ex: '5d'): "))
        else:
            card = self.deck.draw()
        self.community_cards.append(card)

        self.history += ["/"]
        self.history += [str(card)]

        # The person that should play is the first person after the dealer position that is STILL in the game
        self.move_to_next_playing_player(from_position=self.dealer_button_position)
        self.raise_position = self.position_in_play

    def play_river(self):
        self.deck.draw()
        if self.INPUT_CARDS:
            card = Card(input(f"Input the river card (ex: 'Ah'): "))
        else:
            card = self.deck.draw()
        self.community_cards.append(card)

        self.history += ["/"]
        self.history += [str(card)]

        self.move_to_next_playing_player(from_position=self.dealer_button_position)
        self.raise_position = self.position_in_play

    def update_player_balances_at_end_of_stage(self):
        for player in self.players:
            player.player_balance -= player.current_bet
            player.current_bet = 0

    def move_stage_to_total_pot_balance(self):
        self.total_pot_balance += self.stage_pot_balance
        self.stage_pot_balance = 0

    def handle_game_stage(self, action=""):
        if self.game_stage != 1 and self.game_stage != 6:  # nothing to do at start or end of round
            self.play_current_stage(action)

    def move_to_next_game_stage(self, input_cards=None):
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
            self.end_round()
            return

        # If both players are out of balance, it's a showdown until the end
        if self.total_pot_balance == len(self.players) * self.new_player_balance:
            self.move_to_next_game_stage()

    def end_of_round(self):
        return self.game_stage == 6

    def end_round(self):
        self.update_player_balances_at_end_of_stage()
        self.move_stage_to_total_pot_balance()

        if self.count_remaining_players_in_round() > 1:
            self.showdown = True
            evaluator = Evaluator()
            indices_of_potential_winners = []
            for idx, player in enumerate(self.players):
                if player.playing_current_round:
                    indices_of_potential_winners.append(idx)
                    if self.INPUT_CARDS and idx == 1:
                        # Add opponents hand to calculate showdown winner
                        self.players[1].clear_hand()
                        self.players[1].add_card_to_hand(
                            Card(input("Enter the first card from opponent (ex: 5h): "))
                        )
                        self.players[1].add_card_to_hand(
                            Card(input("Enter the second card from opponent (ex: As): "))
                        )
                    hand = CombinedHand(self.community_cards + player.hand)
                    evaluator.add_hands(hand)

            winners = evaluator.get_winner()
            for player in self.players:
                player.playing_current_round = False

            for winner in winners:
                self.players[indices_of_potential_winners[winner]].playing_current_round = True

            for player in self.players:
                if player.is_AI:
                    if player.playing_current_round:
                        player.trash_talk_win()
                    else:
                        player.trash_talk_lose()

        else:
            for player in self.players:
                if player.is_AI:
                    if player.playing_current_round:
                        player.trash_talk_fold()

        self.game_stage = 6  # mark end of round
        self.distribute_pot_to_winning_players()
