# Resources
A non-exhaustive list of repositories, articles and papers I have consulted to put together this project together. To be continually updated.

Git Repositories
- https://github.com/ai-decision/decisionholdem
	-  Recently open-sourced solution, though it seems that it doesn't include the code for abstractions, nor depth-limited solving
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
