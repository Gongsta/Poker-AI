# Kuhn Poker 
Kuhn Poker is a simplified game of poker with 3-cards, invented by Harold E. Kuhn. It is a 2-player zero-sum game.

The CFR implementation is largely taken from this [tutorial](http://modelai.gettysburg.edu/2013/cfr/cfr.pdf), adapted from the Java language to Python.


### Running Code
To run the simulation, simply run this from the `/research/kuhn` directory.
```
python main.py
```

To play against the AI in the terminal, run:
```
python main.py --play
```

### Basic Explanation of Kuhn Poker
- At the beginning of each round, two players each place 1 chip into the pot (we say that they each ante 1 chip)
- Three cards, marked with numbers 1, 2, and 3, are shuffled. One private card is dealt to both players (meaning the other player cannot see the opponent's card)
- Play alternates starting with player 1. On a turn, a player may either **pass** or **bet**.
	- A player that bets places an additional chip into the pot. When a player passes after a bet, the opponent takes all chips in the pot.
	- When there are two successive passes or two successive bets, both players reveal their cards, and the player with the higher card takes all chips in the pot.

Here is a summary of possible play sequences with the resulting chip payoffs:
| Player 1 | Player 2 | Player 1 | Payoff |
| :---: | :---: | :---: | :---: |
| pass | pass | | +1 to player with higher card |
| pass | bet| pass| +1 to player 2|
|pass | bet| bet | +2 to player with higher card|
|bet  | pass| | +1 to player 1 |
| bet  | bet | | +2 to player with higher card|
