from environment import *
from helper import *
import pygame 
import argparse
import os
import time
import joblib

pygame.font.init() # For fonts
pygame.mixer.init() # For sounds

WIDTH, HEIGHT = 900, 500

WIN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Poker By Steven Gong")

WHITE = (255, 255, 255)
BLACK = (25, 25, 25)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
FPS = 60

POKER_BACKGROUND = pygame.transform.scale(pygame.image.load("assets/poker-table.jpg"), (WIDTH, HEIGHT))


FLOP_1_CARD_POSITION = (327,220)
FLOP_2_CARD_POSITION = (374, 220)
FLOP_3_CARD_POSITION = (421,220)
TURN_CARD_POSITION = (468,220)
RIVER_CARD_POSITION = (515,220)

PLAYER_CARD_1 = (398,360)
PLAYER_CARD_2 = (440,360)

OPPONENT_CARD_1 = (398,35)
OPPONENT_CARD_2 = (440,35)


INVERSE_RANK_KEY = {14: "A", 2: "02", 3: "03", 4:"04", 5:"05", 6:"06",
			7:"07", 8:"08", 9:"09", 10:"10", 11: "J", 12: "Q", 13: "K"}
"""Events
post event: pygame.event.post(pygame.event.Event(pygame.USER_EVENT + 1))
And then you check this event in the while True loop

"""

	
dealer_button = pygame.transform.scale(pygame.image.load("assets/dealer_button.png"), (30,30))
card = pygame.image.load("assets/cards/card_hearts_02.png")
CARD_BACK = pygame.image.load("assets/cards/card_back.png")


POT_FONT = pygame.font.SysFont('comicsans', 30)
BET_BUTTON_FONT = pygame.font.SysFont('comicsans', 35)
BET_FONT = pygame.font.SysFont('comicsans', 26)
PLAYERS_FONT = pygame.font.SysFont('comicsans', 24)

# To rescale: pygame.transform.scale(card, (width, height))
# pygame.transform.rotate(card, degrees)

# Pause time: pygame.time.delay(5000)

# rect = Pygame.Rect(x,y, width, height)
# You can then access coordinates with rect.x, rect.y, rect.width, rect.height,

# checking for collisions, use colliderectd

# BUTTONS
fold_rect = pygame.Rect(550, 440, 100,45) 
check_rect = pygame.Rect(665, 440, 100,45)
raise_rect = pygame.Rect(780, 440, 100,45)
buttons = [fold_rect, check_rect, raise_rect]

def load_card_image(card: Card):
	return pygame.image.load("assets/cards/card_" + card.suit.lower() + "_" + INVERSE_RANK_KEY[card.rank] + ".png")

def display_total_pot_balance(env: PokerEnvironment):
	pot_information = POT_FONT.render("Total Pot: $" + str(env.total_pot_balance), 1, WHITE)
	WIN.blit(pot_information, (545, 170))

def display_stage_pot_balance(env: PokerEnvironment):
	pot_information = POT_FONT.render("Current Pot: $" + str(env.stage_pot_balance), 1, WHITE)
	WIN.blit(pot_information, (545, 190))

def display_user_balance(env: PokerEnvironment):
	player_balance = PLAYERS_FONT.render("$"+str(env.players[0].player_balance - env.players[0].current_bet), 1, GREEN)
	WIN.blit(player_balance, (425, 430))

def display_opponent_balance(env: PokerEnvironment):
	opponent_balance = PLAYERS_FONT.render("$" + str(env.players[1].player_balance - env.players[1].current_bet), 1, GREEN)
	WIN.blit(opponent_balance, (425, 108))

def display_user_bet(env: PokerEnvironment):
	pot_information = BET_FONT.render("Bet: $" + str(env.players[0].current_bet), 1, WHITE)
	WIN.blit(pot_information, (420, 320))

def display_opponent_bet(env: PokerEnvironment):
	pot_information = BET_FONT.render("Bet: $" + str(env.players[1].current_bet), 1, WHITE)
	WIN.blit(pot_information, (420, 150))

def display_user_cards(env: PokerEnvironment):
	WIN.blit(load_card_image(env.players[0].hand[0]),PLAYER_CARD_1)
	WIN.blit(load_card_image(env.players[0].hand[1]),PLAYER_CARD_2)

def display_opponent_cards(env: PokerEnvironment):
	WIN.blit(CARD_BACK,OPPONENT_CARD_1)
	WIN.blit(CARD_BACK,OPPONENT_CARD_2)

def reveal_opponent_cards(env: PokerEnvironment):
	WIN.blit(load_card_image(env.players[1].hand[0]),OPPONENT_CARD_1)
	WIN.blit(load_card_image(env.players[1].hand[1]),OPPONENT_CARD_2)

def display_community_cards(env: PokerEnvironment):
	# Draw the CARDS
	for idx, card in enumerate(env.community_cards):
		if idx == 0:
			WIN.blit(load_card_image(card),FLOP_1_CARD_POSITION)
		elif idx == 1:
			WIN.blit(load_card_image(card),FLOP_2_CARD_POSITION)
		elif idx == 2:
			WIN.blit(load_card_image(card),FLOP_3_CARD_POSITION)
		elif idx == 3:
			WIN.blit(load_card_image(card),TURN_CARD_POSITION)
		else:
			WIN.blit(load_card_image(card),RIVER_CARD_POSITION)


def display_dealer_button(env: PokerEnvironment):
	if env.dealer_button_position == 0: # User is the dealer
		WIN.blit(dealer_button, (355, 350))
	else: # Opponent is the dealer
		WIN.blit(dealer_button, (515, 120))

def draw_window(env: PokerEnvironment, god_mode=False, user_input=False):

	WIN.blit(POKER_BACKGROUND, (0,0))

	if env.showdown and env.end_of_round(): # Reveal opponent's cards at showdown
		god_mode = True

	# Display the cards
	display_user_cards(env)
	if god_mode:
		reveal_opponent_cards(env)
	else:
		display_opponent_cards(env)
	
	# Display Community Cards
	display_community_cards(env)

	# Display Pot Information
	display_total_pot_balance(env)
	display_stage_pot_balance(env)
	display_dealer_button(env)	
	
	# TODO: Display Current bet information
	display_user_bet(env)
	display_opponent_bet(env)

	# Display the player names
	AAfilledRoundedRect(WIN, BLACK, pygame.Rect(392,400,120,50), radius=0.7)
	AAfilledRoundedRect(WIN, BLACK, pygame.Rect(392,80,120,50), radius=0.7)
	player_name = PLAYERS_FONT.render("You", 1, WHITE)
	opponent_name = PLAYERS_FONT.render("Opponent",1, GREEN)
	WIN.blit(player_name, (437, 410))
	WIN.blit(opponent_name, (414, 88))
	
	# Display Player Balance Information
	display_user_balance(env)
	display_opponent_balance(env)
	

	# if env.showdown and env.end_of_round(): # Show who won
	if env.end_of_round():
		winning_players = env.get_winning_players_idx()
		if len(winning_players) == 2: # Split the pot
			text = BET_FONT.render("This is a tie", 1, WHITE)
		elif winning_players[0] == 0:
			text = BET_FONT.render("You won!", 1, WHITE)
		else:
			text = BET_FONT.render("You lost.", 1, WHITE)

		WIN.blit(text, (250, 350))

	# Pressable Buttons for Check / Fold / Raise. Only display buttons if it is your turn
	if user_input:
		if env.position_in_play == 0:
			# AAfilledRoundedRect(WIN, RED, pygame.Rect(392,400, 120,50), radius=0.4)
			AAfilledRoundedRect(WIN, RED, fold_rect, radius=0.4)
			AAfilledRoundedRect(WIN, RED, check_rect, radius=0.4)
			AAfilledRoundedRect(WIN, RED, raise_rect, radius=0.4)

			fold_bet = BET_BUTTON_FONT.render("Fold", 1, WHITE)
			WIN.blit(fold_bet, (574, 450))

			if env.min_bet_size == 0:
				check_bet = BET_BUTTON_FONT.render("Check", 1, WHITE) 
				WIN.blit(check_bet, (680, 450))
			else:
				call_bet = BET_BUTTON_FONT.render("Call", 1, WHITE) 
				WIN.blit(call_bet, (689, 450))

			raise_bet = BET_BUTTON_FONT.render("Raise", 1, WHITE)
			WIN.blit(raise_bet, (797, 450))




	pygame.display.update()

def main():
	score = [0, 0] # [PLAYER_SCORE, AI_SCORE]
	# Load the nodeMap
	parser = argparse.ArgumentParser(description="Play Hold'Em Poker against the best AI possible.")
	parser.add_argument("-p", "--play",
                    action="store_true", dest="user_input", default=False,
                    help="Manually play against the AI through a PyGame interface.")
	parser.add_argument("-r", "--replay",
                    action="store_true", dest="replay", default=False,
                    help="replay a history of games")
	parser.add_argument("-g", "--god",
                    action="store_true", dest="god_mode", default=False,
                    help="God mode (see the opponent's cards)")

	args = parser.parse_args()
	user_input = args.user_input
	replay = args.replay
	god_mode = args.god_mode
	
	if replay: # TODO: Load a history of games, and visualize those
		history = joblib.load("HoldemTrainingHistory.joblib")
		game = 0
		game_i = 0
	
	env = PokerEnvironment()
	if user_input or replay:
		env.add_player() # You / replay
	else:
		env.add_AI_player() # Simulation player

	if replay:
		env.add_player() # Player since we want everything to be entered manually
	else:
		env.add_AI_player() # Opponent

	clock = pygame.time.Clock()
	run = True
	while run:
		if user_input or replay: # If you want to render PyGame
			clock.tick(FPS)
		
		handler_called = False
			
		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				run = False
				
			# Check if the buttons are clicked, only process if it is our turn
			if user_input:
				if event.type == pygame.MOUSEBUTTONDOWN and env.position_in_play == 0:
					for i in range(3): # Check for willision with the three buttons
						if buttons[i].collidepoint(pygame.mouse.get_pos()):
							if i == 0:
								env.handle_game_stage("f") # Fold
							elif i == 1:
								env.handle_game_stage("c") # Check / Call
							else:
								env.handle_game_stage("r") # Raise
							
							handler_called = True
							break
		
		if not handler_called:
			if replay:
				print(history[game])
				env.handle_game_stage(history[game][game_i])
				game_i += 1
				if game_i >= len(history[game]): # Move onto the next game
					game += 1
					game_i = 0
					if (game == len(history)): 
						print("Finished replay of all games")
						return

			else:
				env.handle_game_stage()
			
		# At Showdown, reveal opponent's cards and add a delay
		if replay or user_input:
			draw_window(env, god_mode, user_input)

		if user_input and env.end_of_round():
			time.sleep(2)
		
		# if user_input:
		# 	time.sleep(0.2)


	pygame.quit()

if __name__=="__main__":
	main()