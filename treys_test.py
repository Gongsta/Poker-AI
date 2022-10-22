# https://pypi.org/project/treys/
from treys import Evaluator, Deck, Card

evaluator = Evaluator()
deck = Deck()

board = [Card.new("7d"), Card.new("Qh"), Card.new("4h"), Card.new("2s"), Card.new("3c")]
hand = [Card.new("Kc"), Card.new("Qc")]

hand1 = [Card.new("Kc"), Card.new("Qc")]
hand2 = [Card.new("Ts"), Card.new("Th")]
# board = deck.draw(5)
# hand = deck.draw(2)

score = evaluator.evaluate(hand, board)
print(score/ 7462) # 7462 is the WORST POSSIBLE SCORE
# cls = evaluator.get_rank_class(score)
# print(cls)
# print(evaluator.class_to_string(cls))


evaluator.hand_summary(board, [hand1, hand2])