# Poker-AI
A small personal project that I want to work on this summer. First focusing on Heads-Up Hold-Em Poker. Later project will include 8-people in a table poker. 

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
- [ ] Store weights of the trained algorithm
	- Look into `joblib`, `pickle` (slower)
- [ ] Visualize the performance of CFR, and start increasing the amount of cards
- [ ] Create framework to play against the AI that is trained
- [ ] Implement Monte-Carlo CFR for Kuhn Poker
- [ ] Implement Deep CFR for Kuhn Poker


##### 3. Implement the AI for No Limit Texas Hold-Em
- [ ] Implement card abstraction to reduce the game size from $10^{161}$ to $10^{12}$ decision points looking at the Baby Tartanian8 impleeentation
- [ ] Implement Monte-Carlo CFR
- [ ] Implement a distributed version of Monte-Carlo CFR
- [ ] Implement sub-game solving to come up with better strategies during the actual game (key idea presented by Noam Brown. Seach during the game drastically improves performance)
