# Src
This folder holds the main code for the ai logic.
Design decisions I am trying to make:

- Should I call it node or state?  
Ultimately node makes more sense from an intuitive level, because it is a node in the game tree. 
You train strategies on an information set though. You should have infosets in memory, but all possible nodes in memory.
And I  am a very visual person. But ultimately, each node represents a state, so the terms can be used interchangeably.

So I don't have explicit state representation, but rather information sets can be visualized. 
- This is also more efficient for memory right? Actually, you should use states, and optimize with States later. 

Each node only has 1 information set, because the player acting knows what is happening. So it only matters
from the perspective of the opponent?

I am very hesitant about implementing the algortihm with OOP because it tends to be slower. Actually, I just ran an experiment
inside testing/misc_performance.py and it turns out that OOP is faster...?

### Getting Started
Generate the abstractions needed for the game. See `abstraction.py` for the different flags you can pass.
```python
python3 abstraction.py
```
