"""
Wrote this script to illustrate why reinforcement learning doesn't work on RPS, 
(because it outputs a deterministic strategy, which can be easily exploited).
"""
import gym
from gym import spaces
import numpy as np
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import A2C, DQN


ACTION_ARRAY = ["Rock", "Paper", "Scissors"]
class RockPaperScissorsPOMDPEnv(gym.Env):
	"""Custom Environment that follows gym interface
	I am using this to investigate how the RL approach vs. Game Theory approach (finding Nash Equilibrium)
	
	The goal of this simple code is illustrate how things are in the imperfect information world
	
	"""
	metadata = {'render.modes': ['human']}

	def __init__(self):
		super(RockPaperScissorsPOMDPEnv, self).__init__()
		# 1: Rock
		# 2: Paper
		# 3: Scissors
		self.action_space = spaces.Discrete(3) 
		self.observation_space = spaces.Discrete(4) # 0 is used at the beginning state to give an observation
		
		# Used for rendering
		self.done = False
		self.action = 0
		self.reward = 0
		self.opponent_action = 0
	
	def get_opponent_action(self):
		"""
		This is the opponent's strategy for throwing rock paper scissors.
		
		"""
		equal_probability = True
		if equal_probability:
			opponent_action = np.random.randint(1,4)
		else:
			# Sample an action using probabilities
			opponent_action = np.random.choice([1,2,3], p=[0.2, 0.2, 0.6]) 
		return opponent_action
	
	def get_reward(self, action, opponent_action):
		"""
		1 -> Rock
		2 -> Paper
		3 -> Scissors
		action: 1 to 3
		opponent_action: 1 to 3
		
		I use 1-indexing because it's more intuitive for non-programming folks...
		"""
		assert(action >=1 and action <=3)
		assert(opponent_action >=1 and opponent_action <=3)

		if action == opponent_action:
			reward = 0
		elif action == 1:
			if opponent_action == 2: # Rock vs. Paper
				reward = -1
			else: # Rock vs Scissors
				reward = 1
		elif action == 2:
			if opponent_action == 1: # Paper vs. Rock
				reward = 1
			else:  # Paper vs. Scissors
				reward = -1
		elif action == 3:
			if opponent_action == 1: # Scissors vs. Rock
				reward = -1
			else: # Scissors vs. Paper
				reward = 1
		return reward
	def step(self, action):
		"""
		In RPS, we always reach a terminal state in one step.
		Action (we play) -> Environment (opponent plays) -> terminal state, we receive a reward depending on whether we tie, win, or lose
		"""
		opponent_action = self.get_opponent_action()
		action += 1 # We are working in range 1 -> 3, but
		observation = opponent_action
		reward = self.get_reward(action, opponent_action)
			
		done = True
		self.done = done
		self.reward = reward
		self.action = action
		self.opponent_action = opponent_action
		info = {} # No extra info we need
		return observation, reward, done, info

	def reset(self):
		# Used to generate a new episode, in this case we start with an observation of 0 to represent a new state
		self.done = False
		return 0 

	def render(self, mode='human'):
		winner = ""
		if self.reward == 0:
			winner = "Tie"
		elif self.reward == 1:
			winner = "Your AI won, congratulations!"
		else:
			winner = "Your AI lost :("
		if self.done:
			print(ACTION_ARRAY[self.action-1], "vs.", ACTION_ARRAY[self.opponent_action-1], "Result:", winner)

	def close (self):
		pass
	

class RockPaperScissorsMDPEnv(RockPaperScissorsPOMDPEnv):
	"""
	In a fully observable MDP, we can first see the move of the opponent
	"""
	def __init__(self):
		super(RockPaperScissorsMDPEnv, self).__init__()
	
	def step(self, action):
		action += 1 # We are working in range 1 -> 3, but
		observation = self.opponent_action # Not used since we are in a terminal state
		reward = self.get_reward(action, self.opponent_action)
			
		done = True
		self.done = done
		self.reward = reward
		self.action = action
		info = {} # No extra info we need
		return observation, reward, done, info

	def reset(self):
		self.done = False
		self.opponent_action = self.get_opponent_action()
		return self.opponent_action

if __name__ == "__main__":
	env = RockPaperScissorsMDPEnv()
	check_env(env, warn=True)
	# model = DQN('MlpPolicy', env, verbose=1)
	model = A2C('MlpPolicy', env, verbose=1,tensorboard_log="./tensorboard/")
	model.learn(total_timesteps=10000)

	obs = env.reset()
	for i in range(1000):
		action, _state = model.predict(obs)
		obs, reward, done, info = env.step(action)
		env.render()
		if done:
			obs = env.reset()
