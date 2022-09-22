# Poker-AI
A small personal project that I want to work on this summer. First focusing on Heads-Up Hold-Em Poker (i.e. only 2 people playing against each other). Later project will include 8-people in a table poker.

Poker is an interesting game to work on because it is an imperfect information game. This means that unlike perfect-information games such as Chess, in Poker, there is this uncertainty about the opponent's hand, which allows really interesting plays like Bluffing.

Personal notes:
- Naming Convention: "Player" for us, "Opponent" for the enemy
- Future ideas could be using this project as a personal poker trainer (displaying the pot odds). Can help you refine your game


### Basic Explanations of the Game + Class Definitions
Poker is a a family comparing card games in which players wager over which hand is best according to that specific game's rules in ways similar to these rankings. The goal of the game is to win as much money as possible. Unlike other games like Blackjack where players work together and try to beat the "house" (i.e. the casino), Poker is a game in which players try to beat each other.

The version of Poker I am going to be using is **No Limit Texas Hold’Em** (NLTHE), this is by far the most popular one used in all the major tournaments. You can familiarize yourself with the rules [here](https://www.pokernews.com/poker-rules/texas-holdem.htm).


	- Betting rounds happen during each stage At any point, a player can **fold** out of the current round. If only 1 player remains, they win automatically.
    - Then we show 3 community cards, which is known as the **flop**. Followed by a betting round
	- Stage 0-1: Start a new round.
	- Stage 2: Pre-flop.
	- Stage 3: The flop.
    - Stage 4: The turn.
    - Stage 5: River round.

Some of the Class Definitions:
- `Card`s: A Card has a rank (Ace, 2, 3, 4, ..., King) and a suit (Clubs, Diamonds, Hearts, Spades)
- `Deck`: Composed of 52 `Card`s

### Explanation of Directory
- `/assets` contains assets used to run Poker on PyGame
- `/research` contains more background information and code that I wrote before writing the full-on Heads-On Poker 


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
- [ ] Visualize the performance of CFR, and start increasing the amount of cards
- [ ] Create framework to play against the AI that is trained
- [ ] Implement Monte-Carlo CFR for Kuhn Poker
- [ ] Implement Deep CFR for Kuhn Poker


##### 3. Implement the AI for No Limit Texas Hold-Em
- [ ] Calculate pot odds at different stages of the game (use this as a helper when people actually play the game)
- [ ] Implement card abstraction to reduce the game size from $10^{161}$ to $10^{12}$ decision points looking at the Baby Tartanian8 impleeentation
- [ ] Implement Monte-Carlo CFR
- [ ] Implement a distributed version of Monte-Carlo CFR
- [ ] Implement sub-game solving to come up with better strategies during the actual game (key idea presented by Noam Brown. Seach during the game drastically improves performance)
