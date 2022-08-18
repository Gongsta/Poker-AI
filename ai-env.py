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
			# TODO:"opponent_bet": self.env.
			"combined_hand": self.hand # Array

		}

	def _get_info(self): # Useful information for debugging purposes, things outside the current observation
		return {
			"opponent_hand": self.env.players[1].hand
		}


	def _handle_game_stage(self, action=""): # A modification of the PokerEnvironment handle_game_stage() function
		if self.finished_playing_game_stage:
			if self.game_stage != 1:
				self.update_player_balances_at_end_of_stage()
				self.move_stage_to_total_pot_balance()
			self.game_stage += 1
			
			if self.game_stage == 2:
				self.play_preflop()
			elif self.game_stage == 3:
				self.play_flop()
			elif self.game_stage == 4:
				self.play_turn()
			elif self.game_stage == 5:
				self.play_river()
			else:
				if self.game_stage == 6: # We reached the river, and are now in the showdown. We need the evaluator to get the winners, set all losers to playing_current_round false
					evaluator = Evaluator()
					
					indices_of_potential_winners = []
					for idx, player in enumerate(self.players):
						if player.playing_current_round: 
							indices_of_potential_winners.append(idx)
							hand = CombinedHand(self.community_cards + player.hand)
							evaluator.add_hands(hand)
					
					winners = evaluator.get_winner()
					for player in self.players:
						player.playing_current_round = False

					for winner in winners:
						self.players[indices_of_potential_winners[winner]].playing_current_round = True

				self.distribute_pot_to_winning_players()
				self.game_stage = 1
				self.finished_playing_game_stage = False # on the next call of the handler, we will start a new round
		else:
			if self.game_stage == 1:
				self.start_new_round()
			else:
				self.play_current_stage(action)

	def step(self, action):
		# action == 0 -> Fold
		# action == 1 -> Check

		self._update_hand()
		self._handle_game_stage()
        # done = np.array_equal(self._agent_location, self._target_location)
        # reward = 1 if done else 0  # Binary sparse rewards
        observation = self._get_obs()
        info = self._get_info()

        # add a frame to the render collection
        self.renderer.render_step()

        return observation, reward, done, info
		
		
		return observation, reward, done, info
	def reset(self):
		...
		return observation  # reward, done, info can't be included
	def render(self, mode='human'):
		...
	def close (self):
		...