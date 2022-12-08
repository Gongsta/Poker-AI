"""
Design decisions, OOP vs Procedural:
Certain things make natural sense to represent as objects, such as `Player` and `History`. However, what about our functions, such as
the player function, the utility function, or the function that computes and updates the strategy?

We can either treat them as methods of a class (OOP), or a separate function that takes in a class as an argument (procedural).

Let us consider the terminal_utility(), which returns the utility of a terminal history for a particular player. If we choose 
to write this function outside of the class (procedural approach), we could do something like this:

def terminal_utility(i: Player, history: History) -> float:
    if isinstance(history, KuhnHistory):
        #dosomething
    elif isinstance(history, LeducHistory):
        #dosomethingelse

While this way of writing is more consistent with the mathematical notations, the problem that if we add a new game, then we would need to add a new 
elif statement for all of our functions. With OOP, we can simply make `terminal_utility()` an abstract method of the `History` class. Childrens of `History` class 
are then  forced to define the utility function. This approach helps us easily extend to more games, and is the approach I will take below.
"""

# TODO: use NumPy lookup instead of dictionary lookup for drastic speed improvement https://stackoverflow.com/questions/36652533/looking-up-large-sets-of-keys-dictionary-vs-numpy-array


from typing import NewType, Dict, List, Callable, cast
from labml import monit, tracker, logger, experiment
from tqdm import tqdm

CHANCE = "CHANCE_EVENT"

Player = NewType('Player', int)
Action = NewType('Action', str)


class History: 
    """
    The history includes the information about the set of cards we hold.
    
    In our 2 player version, Player 0 always is the first to act (small blind), and player 1 is the second to act (big blind).
    
    """
    def __init__(self):
        pass

    def is_terminal(self):
        raise NotImplementedError()
    
    def actions(self) -> List[Action]:
        raise NotImplementedError()

    def player(self) -> Player:
        # Note, this might returns a chance event possiblity, where you need to implement a chance function
        raise NotImplementedError()

    def is_chance(self):
        raise NotImplementedError()

    def sample_chance_outcome(self):
        # TODO: Determine how to format this API
        raise NotImplementedError()

    def terminal_utility(self, i: Player) -> float:
        assert(self.is_terminal()) # We can only call the utility for a terminal history
        assert(i in [0, 1]) # Only works for 2 player games for now

        raise NotImplementedError()
    
    def __add__(self, action: Action):
        raise NotImplementedError()
    
    def get_infoSet_key(self):
        raise NotImplementedError()
    
    
class InfoSet:
    def __init__(self, key: str):
        self.key = key
        self.regret = {a: 0 for a in self.actions()}
        self.strategy = {a: 0 for a in self.actions()}
        self.cumulative_strategy = {a: 0 for a in self.actions()}
        self.update_strategy()
        assert(sum(self.strategy.values()) == 1.0)

    def actions(self) -> List[Action]:
        raise NotImplementedError()
    
    def player(self) -> Player:
        raise NotImplementedError()

    def to_dict(self):
        return {
            'key': self.key,
            'regret': self.regret,
            'cumulative_strategy': self.cumulative_strategy,
        }
    
    def get_strategy(self):
        """
        Updates the current strategy based on the current regret, using regret matching
        """
        regret = {a: max(r, 0) for a, r in self.regret.items()}
        
        regret_sum = sum(regret.values())
        
        if regret_sum > 0:
            self.strategy = {a: r / regret_sum for a, r in regret.items()}
        else:
            self.strategy = {a: 1/len(self.actions()) for a in self.actions()}
        
    def get_average_strategy(self):
        """
        """
        assert(len(self.actions() == len(self.cumulative_strategy))) # The cumulative strategy should map a probability for every action

        strategy_sum = sum(self.cumulative_strategy.values())
        
        if strategy_sum > 0:
            return {a: s / strategy_sum for a, s in self.cumulative_strategy.items()}
        else:
            return {a: 1/len(self.actions()) for a in self.actions()}
            
        
class CFR:
    def __init__(self, create_infoSet, create_history, n_players: int = 2, iterations: int = 1000):
        self.n_players = n_players
        self.iterations = iterations
        self.infoSets: Dict[str, InfoSet] = {}
        self.create_infoSet = create_infoSet
        self.create_history = create_history

        self.tracker = InfoSetTracker()
    
    def get_infoSet(self, history: History) -> InfoSet:
        infoSet_key = history.get_infoSet_key()
        if infoSet_key not in self.infoSets:
            self.infoSets[infoSet_key] = self.create_infoSet(infoSet_key)

        return self.infoSets[infoSet_key]

    def cfr(self, history: History, i: Player, t: int, pi_0: float, pi_1: float): # Works for two players
        # Return payoff for terminal states
        if history.is_terminal():
            return history.terminal_utility(i)
        elif history.is_chance():
            a = history.sample_chance_outcome() # $\sigma_c$ is simply the $f_c$ function I believe...
            return self.cfr(history + a, i, t, pi_0, pi_1) # Since it is a chance outcome, the player does not change .. TODO: Check logic for this
        
        infoSet = history.get_infoSet()
        assert(infoSet.player() == history.player())
        
        v = 0
        va = {}

        strategy = infoSet.get_strategy()
        for a in infoSet.actions():
            if history.player() == 0:
                va[a] = self.cfr(history + a, i, t, strategy[a] * pi_0, pi_1)
            else:
                va[a] = self.cfr(history + a, i, t, pi_0, strategy[a] * pi_1)

            v += infoSet.strategy[a] * va[a]
        
        if history.player() == i:
            for a in infoSet.actions():
                infoSet.regret[a] += (pi_1 if i == 0 else pi_0) * (va[a] - v)
                # Update cumulative strategy values, this will be used to calculate the average strategy at the end
                infoSet.cumulative_strategy[a] += (pi_0 if i == 0 else pi_1) * infoSet.strategy[a] 
                
            # Update regret matching values
            infoSet.get_strategy()
        
        return v
    
    def solve(self):
        for t in tqdm(range(self.iterations)):
            for player in range(self.n_players):
                self.cfr(self.create_history(), player, t, 1, 1)
            
            tracker.add_global_step()
            self.tracker(self.infoSets)
            tracker.save()
            
            if t % 1000 == 0:
                experiment.save_checkpoint()
            
            logger.inspect(self.infoSets)
        
        
class InfoSetTracker:
    """
    We also want to use this to track exploitability
    """
    def __init__(self):
        tracker.set_histogram(f'strategy.*')
        tracker.set_histogram(f'average_strategy.*')
        tracker.set_histogram(f'regret.*')
    
    def __call__(self, infoSets: Dict[str, InfoSet]):
        for infoSet in infoSets.values():
            avg_strategy = infoSet.get_average_strategy()
            for a in infoSet.actions():
                tracker.add({
                    f'strategy.{infoSet.key}.{a}': infoSet.strategy[a],
                    f'average_strategy.{infoSet.key}.{a}': avg_strategy[a],
                    f'regret.{infoSet.key}.{a}': infoSet.regret[a],
                    })
            