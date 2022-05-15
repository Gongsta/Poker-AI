# Check out pygame implementations
#https://github.com/hsahib2912/AI-Flappy-Birds/blob/b2a72d00913fb0cb0b412d6563f17f6bc3f62b80/AI_project.py
# https://github.com/yenchenlin/DeepLearningFlappyBird/tree/master/game
# Tech with Tim Tutorial: https://www.youtube.com/watch?v=jO6qQDNa2UY&t=4178s&ab_channel=TechWithTim

from environment import *
import pygame 

WIDTH, HEIGHT = 900, 500
WIN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Poker By Steven Gong")

WHITE = (255, 255, 255)
FPS = 60
# env = PokerEnvironment()
# env.add_AI_player()
# env.add_player()

# env.start_new_round()

def draw_window():
	WIN.fill(WHITE)
	pygame.display.update()

def main():
	clock = pygame.time.Clock()
	run = True
	while run:
		
		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				run = False

	pygame.quit()

if __name__=="__main__":
	main()