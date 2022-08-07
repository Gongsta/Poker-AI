# Check out pygame implementations
#https://github.com/hsahib2912/AI-Flappy-Birds/blob/b2a72d00913fb0cb0b412d6563f17f6bc3f62b80/AI_project.py
# https://github.com/yenchenlin/DeepLearningFlappyBird/tree/master/game
# Tech with Tim Tutorial: https://www.youtube.com/watch?v=jO6qQDNa2UY&t=4178s&ab_channel=TechWithTim

from environment import *
import pygame 
import os

pygame.font.init() # For fonts
pygame.mixer.init() # For sounds

WIDTH, HEIGHT = 900, 500

WIN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Poker By Steven Gong")

WHITE = (255, 255, 255)
FPS = 60

POKER_BACKGROUND = pygame.transform.scale(pygame.image.load(""), (900, 500))
"""Events
post event: pygame.event.post(pygame.event.Event(pygame.USER_EVENT + 1))
And then you check this event in the while True loop

"""

# def load_images():
	
# env = PokerEnvironment()
# env.add_AI_player()
# env.add_player()

# env.start_new_round()

card = pygame.image.load("assets/cards/card_hearts_02.png")
# To rescale: pygame.transform.scale(card, (width, height))
# pygame.transform.rotate(card, degrees)

# Pause time: pygame.time.delay(5000)

# rect = Pygame.Rect(x,y, width, height)
# You can then access coordinates with rect.x, rect.y, rect.width, rect.height,

# checking for collisions, use colliderect
def draw_window():
	WIN.blit(POKER_BACKGROUND)
	# WIN.fill(WHITE)
	WIN.blit(card, (200,200))
	

	pygame.display.update()

def main():
	clock = pygame.time.Clock()
	run = True
	while run:
		clock.tick(FPS)
		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				run = False
				
		# keys_pressed = pygame.key.get_pressed()
		draw_window()

	pygame.quit()

if __name__=="__main__":
	main()