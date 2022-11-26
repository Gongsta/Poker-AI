import numpy as np

from copy import copy, deepcopy


class Player:
    def __init__(self):
        self.bets = 1
        self.folded = False
        self.raised = False

    def __repr__(self):
        return f"B: {self.bets} F: {self.folded}"

    def __eq__(self, other):
        return self.bets == other.bets

    def __gt__(self, other):
        return self.bets > other.bets

    def __add__(self, other):
        return self.bets + other.bets

    def __radd__(self, other):
        return self.bets + other

class State:
    def __init__(self, cards, num_players, hand_eval):
        self.num_players = num_players
        self.num_rounds = 1
        self.eval = hand_eval
        self.cards = cards
        self.players = [Player() for _ in range(num_players)]
        self.history = [[] for _ in range(self.num_rounds)]
        self.round = 0
        self.turn = 0
        self.terminal = False

    def __repr__(self):
        return f"{self.history[:self.round+1]}"

    def __eq__(self, other):
        return self.history == other.history and self.cards == other.cards

    def __hash__(self):
        return hash(f'{self.history}, {self.cards}')

    def __copy__(self):
        new_state = State(self.cards, self.num_players, self.eval)
        new_state.players = deepcopy(self.players)
        new_state.history = deepcopy(self.history)
        new_state.turn = self.turn
        new_state.terminal = self.terminal
        new_state.round = self.round

        return new_state

    def info_set(self):
        hole_card = self.cards[self.turn]
        if len(self.cards) > len(self.players):
            board_card = self.cards[len(self.players)]
        else:
            board_card = None

        info_set = f"{hole_card} |{board_card if board_card is not None else ''}| {str(self)}"
        return info_set

    def take(self, action, deep=False):
        if self.terminal == True:
            raise ValueError("Already at a terminal state")

        if deep is True:
            new_state = copy(self)
        else:
            new_state = self

        new_state.history[self.round].append(action)

        curr_player = new_state.players[new_state.turn]
        if action == 'F':
            curr_player.folded = True
            curr_player.raised = False

        elif 'R' in action:
            bet_amount = int(action[:-1])
            call_size = max(self.players).bets - curr_player.bets
            curr_player.bets += bet_amount + call_size
            curr_player.raised = True

        else:
            call_size = max(self.players).bets - curr_player.bets
            curr_player.bets += call_size

        new_state.terminal = new_state.is_terminal()

        return new_state

    def is_terminal(self):
        num_folded = sum([p.folded for p in self.players])

        if num_folded == self.num_players - 1:
            return True

        num_actions = len(self.history[self.round])
        min_actions = self.num_players - num_folded

        max_bet = max([p for p in self.players if p.folded is False],
                        key=lambda k: k.bets)

        end_round = True
        for player in self.players:
            if not player.folded:
                if player < max_bet:
                   end_round = False

        if num_actions >= min_actions and end_round:
            if self.round == self.num_rounds - 1:
                return True
            else:
                self.round += 1
                self.turn = 0
                for p in self.players:
                    p.raised = False

                return False

        self.turn = (self.turn + 1) % self.num_players

        while self.players[self.turn].folded:
            self.turn = (self.turn + 1) % self.num_players

        return False

    def utility(self):
        if len(self.players) - sum([p.folded for p in self.players]) == 1:
            hand_scores = []
            winners = [i for i, _ in enumerate(self.players) if self.players[i].folded == False]

        else:
            board_cards = None if len(self.cards) <= self.num_players else [self.cards[self.num_players]]
            players_in = [i for i, p in enumerate(self.players) if p.folded == False]
            hand_scores = [self.eval(self.cards[i], board_cards) for i in players_in]
            winners = []
            high_score = -1
            for i, score in enumerate(hand_scores):
                if self.players[i].folded == False:
                    if len(winners) == 0 or score > high_score:
                        winners = [i]
                        high_score = score
                    elif score == high_score:
                        winners.append(i)

        pot = sum(self.players)
        payoff = pot / len(winners)
        payoffs = [-p.bets for p in self.players]

        for w in winners:
            payoffs[w] += payoff

        return np.array(payoffs)

    def valid_actions(self):
        any_raises = any([p.raised for p in self.players])
        if any_raises:
            return ['F', 'C']

        return ['F', 'C', '1R']


class Leduc(State):
    def __init__(self, cards, num_players, hand_eval):
        super().__init__(cards, num_players, hand_eval)
        self.num_rounds = 2
        self.players = [Player() for _ in range(num_players)]
        self.history = [[] for _ in range(self.num_rounds)]

    def __copy__(self):
        new_state = Leduc(self.cards, self.num_players, self.eval)
        new_state.players = deepcopy(self.players)
        new_state.history = deepcopy(self.history)
        new_state.turn = self.turn
        new_state.terminal = self.terminal
        new_state.round = self.round

        return new_state


    def valid_actions(self):
        num_raises_so_far = sum([p.raised for p in self.players])

        if num_raises_so_far == self.num_players:
            return ['F', 'C']
        else:
            if self.round == 0:
                return ['F', 'C', '2R']
            else:
                return ['F', 'C', '4R']
