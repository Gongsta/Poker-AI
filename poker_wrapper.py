# Check out pygame implementations
#https://github.com/hsahib2912/AI-Flappy-Birds/blob/b2a72d00913fb0cb0b412d6563f17f6bc3f62b80/AI_project.py
# https://github.com/yenchenlin/DeepLearningFlappyBird/tree/master/game
# Tech with Tim Tutorial: https://www.youtube.com/watch?v=jO6qQDNa2UY&t=4178s&ab_channel=TechWithTim

from environment import *
from helper import *
import pygame 
import os

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


"""Events
post event: pygame.event.post(pygame.event.Event(pygame.USER_EVENT + 1))
And then you check this event in the while True loop

"""

# def load_images():
	
# env = PokerEnvironment()
# env.add_AI_player()
# env.add_player()

# env.start_new_round()

dealer_button = pygame.transform.scale(pygame.image.load("assets/dealer_button.png"), (30,30))
card = pygame.image.load("assets/cards/card_hearts_02.png")
card_back = pygame.image.load("assets/cards/card_back.png")


POT_FONT = pygame.font.SysFont('comicsans', 30)
BET_FONT = pygame.font.SysFont('comicsans', 35)
PLAYERS_FONT = pygame.font.SysFont('comicsans', 24)

# To rescale: pygame.transform.scale(card, (width, height))
# pygame.transform.rotate(card, degrees)

# Pause time: pygame.time.delay(5000)

# rect = Pygame.Rect(x,y, width, height)
# You can then access coordinates with rect.x, rect.y, rect.width, rect.height,

# checking for collisions, use colliderect
def draw_window():
	WIN.blit(POKER_BACKGROUND, (0,0))


	# Draw the CARDS
	WIN.blit(card,FLOP_1_CARD_POSITION)
	WIN.blit(card,FLOP_2_CARD_POSITION)
	WIN.blit(card,FLOP_3_CARD_POSITION)
	WIN.blit(card,TURN_CARD_POSITION)
	WIN.blit(card,RIVER_CARD_POSITION)

	WIN.blit(card, PLAYER_CARD_1)
	WIN.blit(card, PLAYER_CARD_2)

	WIN.blit(card_back,OPPONENT_CARD_1)
	WIN.blit(card_back,OPPONENT_CARD_2)
	
	# Player 1 Information
	AAfilledRoundedRect(WIN, BLACK, pygame.Rect(392,400, 120,50), radius=0.7)
	player_name = PLAYERS_FONT.render("You", 1, WHITE)
	WIN.blit(player_name, (437, 410))
	player_balance = PLAYERS_FONT.render("$1,000", 1, GREEN)
	WIN.blit(player_balance, (425, 430))

	# Opponent
	AAfilledRoundedRect(WIN, BLACK, pygame.Rect(392,80, 120,50), radius=0.7)
	# Text Information
	opponent_name = PLAYERS_FONT.render("Opponent",1, GREEN)
	WIN.blit(opponent_name, (414, 88))
	opponent_balance = PLAYERS_FONT.render("$1,500", 1, GREEN)
	WIN.blit(opponent_balance, (425, 108))
	
	# POT INFORMATION
	pot_information = POT_FONT.render("Pot: $" + str(30), 1, WHITE)
	WIN.blit(pot_information, (415, 170))

	WIN.blit(dealer_button, (515, 120))
	WIN.blit(dealer_button, (355, 350))
	
	
	# Pressable Buttons for Check / Fold / Raise
	# AAfilledRoundedRect(WIN, RED, pygame.Rect(392,400, 120,50), radius=0.4)
	AAfilledRoundedRect(WIN, RED, pygame.Rect(550, 440, 100,45), radius=0.4)
	AAfilledRoundedRect(WIN, RED, pygame.Rect(665, 440, 100,45), radius=0.4)
	AAfilledRoundedRect(WIN, RED, pygame.Rect(780, 440, 100,45), radius=0.4)
	fold_bet = BET_FONT.render("Fold", 1, WHITE)
	check_bet = BET_FONT.render("Check", 1, WHITE) 
	call_bet = BET_FONT.render("Call", 1, WHITE) 
	raise_bet = BET_FONT.render("Raise", 1, WHITE)
	
	WIN.blit(fold_bet, (574, 450))
	WIN.blit(check_bet, (680, 450))
	# WIN.blit(call_bet, (689, 450))
	WIN.blit(raise_bet, (797, 450))


	

	pygame.display.update()

def main():
	clock = pygame.time.Clock()
	run = True
	while run:
		clock.tick(FPS)
		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				run = False
				
			
			# Check if the buttons are clicked
			if event.type == pygame.MOUSEBUTTONDOWN:
				print("key has been pressed")
			
		# keys_pressed = pygame.key.get_pressed()
		draw_window()

	pygame.quit()

if __name__=="__main__":
	main()