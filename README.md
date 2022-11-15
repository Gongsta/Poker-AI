# Poker-AI
Developing an AI agent from scratch to play Heads Up Texas Hold'Em Poker (i.e. 2-player version) using Monte-Carlo Counterfactual Regret Minimization (MCCFR) with Chance Sampling (CS), game abstractions with Earth Mover's Distance, and Depth-Limited Solving (real-time improvement).

**Motivation**: Poker is an interesting game to work on because it is an imperfect information game. This means that unlike perfect-information games such as Chess, in Poker, there is this uncertainty about the opponent's hand, which allows really interesting plays like Bluffing.

I have been incrementally developing the Poker bot, starting with Kuhn Poker which simply implements Vanilla CFR (DONE). Then, with Limit Hold'Em, I needed to use abstractions in order to compute a blueprint strategy. Rules for limit hold'em are explained [here](https://www.pokerlistings.com/limit-texas-holdem). Then, I need to implement depth-limited solving, since real-time solving is a key component in creating a superhuman AI (seen for example for AlphaZero).

I might need to write C++ code, since training on the Python code might take a very long time.

### Installation
This repository requires Python>=3.9.


### Basic Explanations of the Game + Class Definitions
Poker is a a family comparing card games in which players wager over which hand is best according to that specific game's rules in ways similar to these rankings. The goal of the game is to win as much money as possible. Unlike other games like Blackjack where players work together and try to beat the "house" (i.e. the casino), Poker is a game in which players try to beat each other.

The version of Poker I am going to be using is **No Limit Texas Hold’Em** (NLTHE), this is by far the most popular one used in all the major tournaments. You can familiarize yourself with the rules [here](https://www.pokernews.com/poker-rules/texas-holdem.htm).

Some Specifications
- Betting rounds happen during each stage. At any point, a player can **fold** out of the current round. If only 1 player remains, they win automatically.
- Then we show 3 community cards, which is known as the **flop**. Followed by a betting round
- Stage 0-1: Start a new round.
- Stage 2: Pre-flop.
- Stage 3: The flop.
- Stage 4: The turn.
- Stage 5: River round.

Class Definitions
- `Card`s: A Card has a rank (Ace, 2, 3, 4, ..., King) and a suit (Clubs, Diamonds, Hearts, Spades)
- `Deck`: Composed of 52 `Card`s

### Explanation of Directory
- `/assets` contains assets used to run Poker on PyGame
- `/research` contains more background information and code that I wrote before writing the full-on Heads-On Poker


### High-Level overview of how this AI tries to beat Poker
The main idea of the Poker algorithm is storing a strategy profile for "every possible scenario" to maximize our reward at all stages of a poker round. In Poker No-Limit Hold'Em has, this means we would store $10^{161}$ sets of
strategy profiles. That is simply not feasible. Hence, the first step is to find a way to abstract scenarios.

There are two common types of abstraction: information (i.e. cards) abstraction and action (i.e. bet size) abstraction. 

For action abstraction, I have decided to simplify the actions to fold (f), check (k), call (c), small-bet (0.5x pot), medium-bet (1x pot), large-bet (2x pot), and all-in. 

Card Abstractions are done by grouping hands with similar equity distributions into the same bucket/cluster/node. Equity is a measure of expected hand strength, which is your probability of winning given a uniform random rollout of community cards and random opponent private cards.The idea is based from this [paper](https://www.cs.cmu.edu/~sandholm/potential-aware_imperfect-recall.aaai14.pdf), which talks about potential-aware and distribution aware card abstractions. In this project, cards are bucketed in 169 clusters (pre-flop), TBD (flop), TBD (turn), and TBD (river). 

### Concepts
##### Evaluating Performance
How to evaluate the performance of the Poker AI? There are 3 main ways to measure the quality of a poker agent:
1. Measure it against other poker agents
	- This is done through `slumbot/slumbot_api.py`
2. Measure it against humans
3. Measure it against a “best response” agent that always plays the best response to see how well the agent does in the worst case

### Timeline / Goals 
##### 1. Create a basic Poker Environment to play in
- [x] Write classes for `Card`, `Deck`, `Player`, `PokerEnvironment`
- [x] Write a PyGame interface for users to play against the AI
	- [ ] Highlight the cards of the best hand, just like in a real online poker environment

##### 2. Learn to write the AI
- [x] Explore different reinforcement learning algorithms, look into what papers have done
	- Realized that RL algorithms don't work at all in imperfect information games. It fails at the simplest game of Rock-Paper-Scissors because the policy it comes up with is deterministic, and easily exploitable. What we need to do is take a game theory approach, using the idea of Counterfactual Regret Minimization (CFR) to create a strategy that converges to the Nash Equilibrium.
- [x] Implement regret-matching for Rock-Paper-Scissors
- [x] Write the vanilla CFR code for Kuhn Poker
- [x] Store weights of the trained algorithm
	- Look into `joblib`, `pickle` (slower)
- [x] Visualize the performance of CFR, and start increasing the amount of cards
- [x] Create framework to play against the AI that is trained
- [x] Implement Monte-Carlo CFR for Kuhn Poker
- [x] Implement Deep CFR for Kuhn Poker

##### 3. Implement the AI for No Limit Texas Hold-Em
- [ ] Implement abstraction, generate a table for pre-flop using Monte-Carlo technique, same card representation should have exactly the same EHS
- [ ] Calculate pot odds at different stages of the game (use this as a helper when people actually play the game)
- [ ] Implement card abstraction to reduce the game size from $10^{161}$ to $10^{12}$ decision points 
	- [ ] Implement basic monte-carlo method to calculate the EHS of a pair of cards at different stages of the game. This assumes a random uniform draw of opponent hands and random uniform rollout of public cards
	- [ ] Implement a simple clustering algorithm that uses these EHS to cluster various cards / scenarios together...?
	- [ x ] Implement parallel methods with `joblib` to accelerate speed: Went from ~0.17s per hand to ~0.05s per hand
	- Use 169, 5000, 5000, 5000 buckets as outlined here: https://www.cs.cmu.edu/~sandholm/potential-aware_imperfect-recall.aaai14.pdf
- [ ] Implement Monte-Carlo CFR
- [ ] Implement a distributed version of Monte-Carlo CFR
- [ ] Implement sub-game solving to come up with better strategies during the actual game (key idea presented by Noam Brown. Seach during the game drastically improves performance)... I don't know how to do this


##### 4. Future Steps
- [ ] Implement Computer Vision + Deep Learning to recognize Poker cards, and so you can deploy this model in real life by mounting a camera to your head.
- [ ] Use this project as a personal poker trainer (displaying the pot odds). Can help you refine your game