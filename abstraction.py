"""
Python file that takes care of card abstractions for Poker. 

Inspired from Noam Brown's paper: https://arxiv.org/pdf/1805.08195.pdf

"The blueprint abstraction treats every poker hand separately on the first betting round (where there are
169 strategically distinct hands). On the remaining betting rounds, the hands are grouped into 30,000
buckets. The hands in each bucket are treated identically and have a shared strategy, so
they can be thought as sharing an abstract infoset".
"""

# Preflop Abstraction
"""
For the Pre-flop, we can make a lossless abstraction with exactly 169 buckets. The idea here is that what specific suits
our private cards are doesn't matter. The only thing that matters is whether both cards are suited or not.

This is how the number 169 is calculated:
- For cards that are not pocket pairs, we have (13 choose 2) = 13 * 12 / 2 = 78 buckets (since order doesn't matter)
- These cards that are not pocket pairs can also be suited, so we must differentiate them. We have 78 * 2 = 156 buckets
- Finally, for cards that are pocket pairs, we have 13 extra buckets (Pair of Aces, Pair of 2, ... Pair Kings). 156 + 13 = 169 buckets

Note that a pair cannot be suited, so we don't need to multiply by 2.
"""

# Flop Abstraction
"""
Abstracting other game stages is more complicated. Here, we use 

"""


# Turn Abstraction

# River Abstraction
