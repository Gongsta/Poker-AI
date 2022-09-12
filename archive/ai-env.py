import gym
from gym import spaces
import numpy as np
from environment import PokerEnvironment
from evaluator import CombinedHand

N_DISCRETE_ACTIONS = 3 # Fold, Check/Bet, Raise

"""
Resources to learn to write the gym env:
- https://www.gymlibrary.ml/content/environment_creation/
- https://stable-baselines3.readthedocs.io/en/master/guide/custom_env.html
- https://colab.research.google.com/github/araffin/rl-tutorial-jnrr19/blob/master/5_custom_gym_env.ipynb?authuser=0#scrollTo=i62yf2LvSAYY

"""
class PokerEnv(gym.Env):
	"""Custom Environment that follows gym interface"""
	metadata = {'render.modes': ['human']}

	def __init__(self):
		super(PokerEnv, self).__init__()
		
		self.hand = []

		# My own poker interface, but slightly modified logic to let the AI Agent
		self.env = PokerEnvironment() 
		self.env.add_player()
		self.env.add_AI_player()

		self.action_space = spaces.Discrete(N_DISCRETE_ACTIONS) # 1. Fold 2. Check/Bet 3.Raise
		# Reduce the observation space
		self.observation_space = spaces.Dict(
			{
				"pot_balance": spaces.Discrete(50), # Pot bet between 0...50 BB
				"dealer_position": spaces.Discrete(2), # 0 or 1 depending on whether we are the dealer, this is important for aggression
				# "player_bet": spaces.Discrete(50), #represents how much the player has already put in the pot
				"opponent_bet": spaces.Discrete(50), # This is what the opponent bets, so you need to match it, so that would be your reward
				"combined_hand": spaces.MultiBinary(52) # Array of 52 bits where each bit will be either 0 or 1
			}
		)


	def _update_hand(self):
		self.hand = CombinedHand(self.env.players[0].hand + self.env.community_cards).get_array_binary_representation()
	
	def _get_obs(self):
		return {
			"pot_balance": self.env.total_pot_balance,
			"opponent_bet": self.env.stage_pot_balance, # TODO: Will need to be changed for raises and stuff. Since this is just one turn, and no raises 
			"combined_hand": self.hand # Array

		}

	def _get_info(self): # Useful information for debugging purposes, things outside the current observation
		return {
			"opponent_hand": self.env.players[1].hand
		}

	def step(self, action):
		"""
		There are two cases:
		1. We are the dealer, in which case the opponent plays first
		2. The opponent is the dealer, in which ase we play first
		
		The idea is this: We play, and then get an observation on how the opponent reacts based on how we play.
		However, if we go second, our observation will be based on the fact that there is nothing put on the pot, and action is on us.
		"""
		# action == 0 -> Fold
		# action == 1 -> Check

		self._update_hand()
		if self.env.dealer_button_position == 0: # Action is on the opponent first
			self.env.handle_game_stage()
		
		else: # Action is on us first
        # done = np.array_equal(self._agent_location, self._target_location)
		if self.env.game_stage == 1:
        reward = 
        observation = self._get_obs()
        info = self._get_info()

        # add a frame to the render collection
        self.renderer.render_step()

        return observation, reward, done, info
		
	def reset(self):
		return observation  # reward, done, info can't be included

	def render(self, mode='human'):
		...
	def close (self):
		...