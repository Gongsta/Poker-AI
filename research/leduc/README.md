# Leduc Hold'Em Poker
This was taken from the original repository https://github.com/zanussbaum/pluribus, and has been modified, annotated for my personal understanding, and improved. 

### Basic Rules of Leduc Poker
In Leduc hold’Em, the deck consists of two suits with cards 1,2 and 3.
- At the beginning of the round, a single private card is dealt to each player. Each of them put 1 chip into the pot.  
- A round of bet follows, with a maximum raise of 2
- A community card is revealed, followed by a second and final round of betting. The max raise rize is 4 this time. At showdown, pairs beat high cards.

## CFR and MCCFR


MCCFR is a variant of CFR where instead of traversing the whole game tree, we will only sample actions so that we will traverse paths on the game tree that are more likely than others. This makes MCCFR a much efficient algorithm for large games.

## How To Use 

`python [vanilla.py|montey.py]` 
to run the CFR/MCCFR variant

`python search.py` to play a game of Leduc. 

CFR converges in around ~10,000 iterations.

MCCFR can converge in around ~10,000, but is more stable around ~20,000 iterations.

For Leduc, you need aorund 50,000 iterations. 