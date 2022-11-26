class Card:
    """Inspired from pycfr card.py"""
    SUIT_STRING = {
        1: "s",
        2: "h",
        3: "d",
        4: "c"
    }
    CARD_STRING = {
        2: "2",
        3: "3",
        4: "4",
        5: "5",
        6: "6",
        7: "7",
        8: "8",
        9: "9",
        10: "T",
        11: "J",
        12: "Q",
        13: "K",
        14: "A"
    }
    def __init__(self, rank, suit):
        self.rank = rank
        self.suit = suit

    def __repr__(self):
        return '{}{}'.format(self.CARD_STRING[self.rank], self.SUIT_STRING[self.suit])

    def __eq__(self, card):
        return card.rank == self.rank

    def __lt__(self, card):
        return self.rank < card.rank

    def __hash__(self):
        return hash(repr(self))



