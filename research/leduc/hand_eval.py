def kuhn_eval(card, public):
    return card.rank

def leduc_eval(hole_card, board):
    cards = [hole_card] + board

    if cards.count(hole_card) > 1:
        return 15*14 + hole_card.rank

    return 14 * max(cards).rank + min(cards).rank
