# Poker-AI
Developing an AI agent from scratch to play Heads Up Texas Hold'Em Poker (i.e. 2-player version) using Monte-Carlo Counterfactual Regret Minimization (MCCFR) with Chance Sampling (CS), game abstractions with using K-Means Clustering, and Depth-Limited Solving (real-time improvement).

**Motivation**: Poker is an interesting game to work on because it is an imperfect information game. This means that unlike perfect-information games such as Chess, in Poker, there is this uncertainty about the opponent's hand, which allows really interesting plays like Bluffing.

I have been incrementally developing the Poker bot, starting with Kuhn Poker which simply implements Vanilla CFR (DONE). Then, with Limit Hold'Em, I needed to use abstractions in order to compute a blueprint strategy. Rules for limit hold'em are explained [here](https://www.pokerlistings.com/limit-texas-holdem). Then, I need to implement depth-limited solving, since real-time solving is a key component in creating a superhuman AI (seen for example for AlphaZero).

I might need to write C++ code, since training on the Python code might take a very long time.

Also a blog series will be coming out to explain how this codebase works.

### Installation
This repository requires Python>=3.9.
```bash
pip install -r requirements.txt
```

Originally did the game in PyGame, but it's not responsive so I don't really like that. I plan on moving it to flask.


### Basic Explanations of the Game + Class Definitions
Poker is a family of comparing card games in which players wager over which hand is best according to that specific game's rules in ways similar to these rankings. The goal of the game is to win as much money as possible. Unlike other games like Blackjack where players work together and try to beat the "house" (i.e. the casino), Poker is a game in which players try to beat each other.

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
The main idea of the Poker algorithm is storing a strategy profile for "every possible scenario" to maximize our reward at all stages of a poker round. 
- A "Strategy profile" means defining how the AI should behave. For example, if you had pocket aces on the pre-flop, and you are the big blind, a strategy profile might tell you that you should make a pot-size raise 50% of the time, and go all-in the other 50% of the time.

However, Poker No-Limit Hold'Em is an extremely large game. This means we would have to store $10^{161}$ sets of strategy profiles. That is simply not feasible. Hence, the first step is to find a way to group scenarios together so there are less scenarios to keep track off. We are thus solving a smaller game.

After we've created an abstract version of the game that is smaller, we solve it directly using an algorithm called Counterfactual Regret Minimization (CFR).

Finally, we implement depth-limited solving, which is a real-time search algorithm invented in 2018 that drastically improves the performance of the AI and helps it make better decisions and have it be less vulnerable.


#### Step 1: Abstraction
There are two common types of abstraction: information (i.e. cards) abstraction and action (i.e. bet size) abstraction.

For action abstraction, you size the bets according to how much money is currently being put into the pot. See the `holdem.py` python file for more details.

Card Abstractions are done by grouping hands with similar equity distributions into the same bucket/cluster/node. Equity is a measure of expected hand strength, which is your probability of winning given a uniform random rollout of community cards and random opponent private cards.The idea is based from this [paper](https://www.cs.cmu.edu/~sandholm/potential-aware_imperfect-recall.aaai14.pdf), which talks about potential-aware and distribution aware card abstractions. In this project, cards are bucketed in 169 clusters (pre-flop), TBD (flop), TBD (turn), and TBD (river). 


Slumbot abstractions:
https://nanopdf.com/queue/slumbot-nl-solving-large-games-with-counterfactual_pdf?queue_id=-1&x=1670505293&z=OTkuMjA5LjUyLjEzOA==

#### Step 2: Generate a Blueprint Strategy with CFR
Counterfactual Regret Minimization (CFR) is a self-play algorithm used to learn a strategy to a game by repeatedly playing a game and updating its strategy to improve how much it "regret" taking a decision at a each decision point. This strategy has been wildly successful for imperfect information games like poker. 

[CFR](https://www.quora.com/What-is-an-intuitive-explanation-of-counterfactual-regret-minimization) has been shown to converge to Nash Equilibrium strategy for 2 player zero-sum games. 

[Here](http://modelai.gettysburg.edu/2013/cfr/cfr.pdf) is a great resource for the curious reader.

##### Step 3: Real-time Solving with Depth-Limited Search
To be implemented. The idea is that we can improve our strategies in realtime.

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
- [x] Implement abstraction, generate a table for pre-flop using Monte-Carlo technique, same card representation should have exactly the same EHS
- [x] Calculate pot odds at different stages of the game (use this as a helper when people actually play the game)
- [x] Implement card abstraction to reduce the game size from $10^{161}$ to $10^{12}$ decision points 
	- [x] Implement basic monte-carlo method to calculate the EHS of a pair of cards at different stages of the game. This assumes a random uniform draw of opponent hands and random uniform rollout of public cards
	- [x] Implement a simple clustering algorithm that uses these EHS to cluster various cards / scenarios together...?
	- [ x ] Implement parallel methods with `joblib` to accelerate speed: Went from ~0.17s per hand to ~0.05s per hand
	- Use 169, 5000, 5000, 5000 buckets as outlined here: https://www.cs.cmu.edu/~sandholm/potential-aware_imperfect-recall.aaai14.pdf
		- Hmm maybe this is too much
- [ ] Implement helper functions to calculate expected values of various strategies (utilities of a strategy profile)
- [ ] Implement Monte-Carlo CFR
- [ ] Implement a distributed version of Monte-Carlo CFR
- [ ] Implement sub-game solving to come up with better strategies during the actual game (key idea presented by Noam Brown. Seach during the game drastically improves performance)... I don't know how to do this


##### 4. Future Steps
- [ ] Implement Computer Vision + Deep Learning to recognize Poker cards, and so you can deploy this model in real life by mounting a camera to your head.
- [ X ] Use this project as a personal poker trainer (displaying the pot odds). Can help you refine your game, see the `learn_pot_odds.py` file
- [ X ] release CFR code as a library, since there is no universal support of CFR. I wish the researchers released those, but everyone seems to just do their own thing. It kind of seems like the early days of neural networks, when everyone would write their own backward pass for backpropagation, until Tensorflow and Pytorch came along.


## Resources
A non-exhaustive list of repositories, articles and papers I have consulted to put together this project together. To be continually updated.
Git Repositories
-  Recently open-sourced solution, though it seems that it doesn't include the code for abstractions, nor depth-limited solving https://github.com/ai-decision/decisionholdem
- https://github.com/matthewkennedy5/Poker -> Really good writing
- https://github.com/fedden/poker_ai
- https://github.com/jneckar/skybet
- https://github.com/zanussbaum/pluribus (An attempt at implementing Pluribus)
- https://github.com/doas3140/PyStack (Python Implementation of DeepStack)
- These students tried to make a copy of Libratus: https://github.com/michalp21/coms4995-finalproj
	https://github.com/tansey/pycfr (8 years old) -> implementation in CFR, not support nolimit texas holdem
- Pokerbot https://github.com/dickreuter/Poker
- Gym Environment https://github.com/dickreuter/neuron_poker

Blogs
- https://int8.io/counterfactual-regret-minimization-for-poker-ai/
- https://aipokertutorial.com/

Other:
- Really good [tutorial](https://aipokertutorial.com/) by a guy who played 10+ years online poker
- Poker Mathematics [Book](http://www.pokerbooks.lt/books/en/The_Mathematics_of_Poker.pdf)

Paper links
- [An Introduction to CFR](http://modelai.gettysburg.edu/2013/cfr/cfr.pdf) (Neller, 2013) ESSENTIAL
- **Vanilla CFR** 
	- (CFR first introduced) [Regret Minimization in Games with Incomplete Information](https://poker.cs.ualberta.ca/publications/NIPS07-cfr.pdf) (Bowling, 2007)
	- [Using CFR to Create Competitive Multiplayer Poker Agents](https://poker.cs.ualberta.ca/publications/AAMAS10.pdf) (Risk, 2010)
- [Efficient MCCFR in Games with Many Player Actions](https://proceedings.neurips.cc/paper/2012/file/3df1d4b96d8976ff5986393e8767f5b2-Paper.pdf) (Burch, 2012)
- **CFR-BR** (CFR-Best Response)
	- [Finding Optimal Abstract Strategies in Extensive-Form Games](https://poker.cs.ualberta.ca/publications/AAAI12-cfrbr.pdf) (Burch, 2012) (IMPORTANT paper in finding)
- **Monte-Carlo CFR** (IMPORTANT)
- **CFR-D (Decomposition)**
	- [Solving Imperfect Information Games Using Decomposition](https://poker.cs.ualberta.ca/publications/aaai2014-cfrd.pdf) (Burch, 2013) 
- **CFR+**
	- (Pseudocode) [Solving Large Imperfect Information Games Using CFR+](https://arxiv.org/pdf/1407.5042.pdf) (Tammelin, 2014) 
	- [Solving Heads-up Limit Texas Hold’em](https://poker.cs.ualberta.ca/publications/2015-ijcai-cfrplus.pdf) (Tammelin, 2015)
- **RBP** Regret-Based Pruning 
	- RBP is particularly useful in large games where many actions are suboptimal, but where it is not known beforehand which actions those are
	- [Regret-Based Pruning in Extensive-Form Games](https://www.cs.cmu.edu/~noamb/papers/15-NIPS-Regret-Based.pdf) (Brown, 2015)
- Warm Start CFR
	- [Strategy-Based Warm Starting for Regret Minimization in Games](https://www.cs.cmu.edu/~noamb/papers/16-AAAI-Strategy-Based.pdf) (Brown, 2015)
- **DCFR** (Discounted CFR)
	- [Solving Imperfect-Information Games via Discounted Regret Minimization](https://arxiv.org/abs/1809.04040) (Brown, 2018)
- **ICFR** (instant CFR)
	- [Efficient CFR for Imperfect Information Games with Instant Updates](https://realworld-sdm.github.io/paper/27.pdf) (Li, 2019)
- **Deep CFR**
	- [Deep Counterfactual Regret Minimization](https://arxiv.org/abs/1811.00164) (Brown, 2018)
	- [Combining Deep Reinforcement Learning and Search for Imperfect-Information Games](https://arxiv.org/abs/2007.13544) (Brown, 2020)


Other ideas
- **Depth-Limited Solving** (IMPORTANT): This is a key technique that allows us to train a top tier Poker AI on our local computer, by improving a blueprint strategy.
	- [Depth-Limited Solving for Imperfect-Information Games](https://arxiv.org/pdf/1805.08195.pdf) (Brown, 2018)
- **Abstractions** (IMPORTANT):  See [[Game Abstraction]]. Abstractions are absolutely necessary, since Texas Hold'Em is too big to solve directly 
	- [A heads-up no-limit Texas Hold’em poker player: Discretized betting models and automatically generated equilibrium-finding programs](https://www.cs.cmu.edu/~sandholm/tartanian.AAMAS08.pdf)
	- [Action Translation in Extensive-Form Games with Large Action Spaces: Axioms, Paradoxes, and the Pseudo-Harmonic Mapping](https://www.cs.cmu.edu/~sandholm/reverse%20mapping.ijcai13.pdf) (Sandholm, 2013)
	- [Evaluating State-Space Abstractions in Extensive-Form Games](https://poker.cs.ualberta.ca/publications/AAMAS13-abstraction.pdf) (Burch, 2013)
	- [Potential-Aware Imperfect-Recall Abstraction with Earth Mover’s Distance in Imperfect-Information Games](https://www.cs.cmu.edu/~sandholm/potential-aware_imperfect-recall.aaai14.pdf) (Sandholm, 2014)
	- [Abstraction for Solving Large Incomplete-Information Games](https://www.cs.cmu.edu/~sandholm/game%20abstraction.aaai15SMT.pdf) (Sandholm, 2015)
	- [Hierarchical Abstraction, Distributed Equilibrium Computation, and Post-Processing, with Application to a Champion No-Limit Texas Hold’em Agent](https://www.cs.cmu.edu/~noamb/papers/15-AAMAS-Tartanian7.pdf) (Brown, 2015)
- Subgame Solving: This seems to be impossible to do on a local computer
	- [Safe and Nested Subgame Solving for Imperfect-Information Games](https://arxiv.org/abs/1705.02955) (Brown, 2017)
- Measuring the Size of Poker
	- [Measuring the Size of Large No-Limit Poker Games](https://arxiv.org/pdf/1302.7008.pdf) (Johnson, 2013)
- Evaluating the Performance of a Poker Agent
	- [A TOOL FOR THE DIRECT ASSESSMENT OF POKER DECISIONS](https://poker.cs.ualberta.ca/publications/divat-icgaj.pdf) (Billings, 2006) 
	- [Strategy Evaluation in Extensive Games with Importance Sampling](https://poker.cs.ualberta.ca/publications/ICML08.pdf) (Bowling, 2008)

Poker Equity: https://www.pokernews.com/strategy/talking-poker-equity-21291.htm#:~:text=When%20you%20play%20poker%2C%20'Equity,at%20that%20moment%20is%20%2490.

Other Links (Web Pages + Videos)
- https://poker.cs.ualberta.ca/resources.html, this is really good https://poker.cs.ualberta.ca/general_information.html for general information
- Poker Database: https://poker.cs.ualberta.ca/irc_poker_database.html
- [The State of Techniques for Solving Large Imperfect-Information Games, Including Poker](https://www.youtube.com/watch?v=QgCxCeoW5JI&ab_channel=MicrosoftResearch) by Sandholm, really solid overview about abstractions of the game
- [Superhuman AI for heads-up no-limit poker: Libratus beats top professionals](https://www.youtube.com/watch?v=2dX0lwaQRX0&t=2591s&ab_channel=NoamBrown) by Noam Brown
	- [AI for Imperfect-Information Games: Beating Top Humans in No-Limit Poker](https://www.youtube.com/watch?v=McV4a6umbAY&ab_channel=MicrosoftResearch) by Noam Brown at Microsoft Research

Poker Agents Papers
- Slumbot "250,000 core hours and 2 TB of RAM to compute its strategy"
- [Polaris](https://www.ifaamas.org/Proceedings/aamas09/pdf/06_Demos/d_11.pdf) (2008)
- [Baby Tartanian 8](https://www.cs.cmu.edu/~sandholm/BabyTartanian8.ijcai16demo.pdf) (2016) "2 million core hours and 18 TB of RAM to compute its strategy"
- [DeepStack](https://static1.squarespace.com/static/58a75073e6f2e1c1d5b36630/t/58b7a3dce3df28761dd25e54/1488430045412/DeepStack.pdf) (2017)
- [Libratus](https://www.cs.cmu.edu/~noamb/papers/17-IJCAI-Libratus.pdf) (2017)
	1. Blueprint Strategy (Full-Game Strategy) using MCCFR
	2. Subgame Solving with CFR+
	3. Adapt to opponent
- [Pluribus](https://www.cs.cmu.edu/~noamb/papers/19-Science-Superhuman.pdf), video [here](https://www.youtube.com/watch?v=u90TbxK7VEA&ab_channel=TwoMinutePapers) (2019)