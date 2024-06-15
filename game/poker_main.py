import sys

sys.path.append("../src")

from environment import *
from helper import *
import pygame
import argparse
import time
import joblib

pygame.font.init()  # For fonts
pygame.mixer.init()  # For sounds

SCALE = 1
WIDTH, HEIGHT = 1280, 720
# WIDTH, HEIGHT = 1289, 791
# WIDTH, HEIGHT = 1600, 900

WIN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Poker By Steven Gong")


WHITE = (255, 255, 255)
BLACK = (25, 25, 25)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
FPS = 60

POKER_BACKGROUND = pygame.transform.scale(
    pygame.image.load("assets/poker-table.png"), (WIDTH, HEIGHT)
)

FLOP_1_CARD_POSITION = (400, HEIGHT / 2 - 65)
FLOP_2_CARD_POSITION = (490, HEIGHT / 2 - 65)
FLOP_3_CARD_POSITION = (580, HEIGHT / 2 - 65)
TURN_CARD_POSITION = (670, HEIGHT / 2 - 65)
RIVER_CARD_POSITION = (760, HEIGHT / 2 - 65)

PLAYER_CARD_1 = (550, HEIGHT - 220)
PLAYER_CARD_2 = (600, HEIGHT - 220)

OPPONENT_CARD_1 = (550, 35)
OPPONENT_CARD_2 = (600, 35)


INVERSE_RANK_KEY = {
    14: "A",
    2: "02",
    3: "03",
    4: "04",
    5: "05",
    6: "06",
    7: "07",
    8: "08",
    9: "09",
    10: "10",
    11: "J",
    12: "Q",
    13: "K",
}
"""Events
post event: pygame.event.post(pygame.event.Event(pygame.USER_EVENT + 1))
And then you check this event in the while True loop

"""


dealer_button = pygame.transform.scale(pygame.image.load("assets/dealer_button.png"), (30, 30))
CARD_BACK = pygame.transform.scale(pygame.image.load("../assets/back.png"), (263 / 3, 376 / 3))


POT_FONT = pygame.font.SysFont("Roboto", 30, bold=True)
BET_BUTTON_FONT = pygame.font.SysFont("Roboto", 24, bold=True)
BET_FONT = pygame.font.SysFont("Roboto", 26, bold=True)
PLAYERS_FONT = pygame.font.SysFont("Roboto", 24, bold=True)

# To rescale: pygame.transform.scale(card, (width, height))
# pygame.transform.rotate(card, degrees)

# Pause time: pygame.time.delay(5000)

# rect = Pygame.Rect(x,y, width, height)
# You can then access coordinates with rect.x, rect.y, rect.width, rect.height,

# checking for collisions, use colliderectd

# BUTTONS
fold_rect = pygame.Rect(800, HEIGHT - 80, 80, 45)
check_rect = pygame.Rect(887, HEIGHT - 80, 100, 45)  # Can also be call button
custom_rect = pygame.Rect(995, HEIGHT - 80, 80, 45)
buttons = [fold_rect, check_rect, custom_rect]

input_box = pygame.Rect(1060, HEIGHT - 80, 140, 45)
color_inactive = pygame.Color("lightskyblue3")
color_active = pygame.Color("dodgerblue2")
color = color_inactive
active = False
input_bet_text = ""
warning_text = ""
done = False

cursor_counter = 0

def load_card_image(card: Card):
    # 263 × 376
    return pygame.transform.scale(
        pygame.image.load("../assets/" + str(card) + ".png"), (263 / 3, 376 / 3)
    )

def display_total_pot_balance(env: PokerEnvironment):
    pot_information = POT_FONT.render("Total Pot: $" + str(env.total_pot_balance), 1, WHITE)
    WIN.blit(pot_information, (900, HEIGHT / 2 - 30))


def display_stage_pot_balance(env: PokerEnvironment):
    pot_information = POT_FONT.render("Current Pot: $" + str(env.stage_pot_balance), 1, WHITE)
    WIN.blit(pot_information, (900, HEIGHT / 2))


def display_user_balance(env: PokerEnvironment):
    player_balance = PLAYERS_FONT.render(
        "$" + str(env.players[0].player_balance - env.players[0].current_bet), 1, GREEN
    )
    WIN.blit(player_balance, (WIDTH / 2 + 80, HEIGHT - 200))


def display_opponent_balance(env: PokerEnvironment):
    opponent_balance = PLAYERS_FONT.render(
        "$" + str(env.players[1].player_balance - env.players[1].current_bet), 1, GREEN
    )
    WIN.blit(opponent_balance, (WIDTH / 2 + 80, 100))


def display_user_bet(env: PokerEnvironment):
    pot_information = BET_FONT.render("Bet: $" + str(env.players[0].current_bet), 1, WHITE)
    WIN.blit(pot_information, (WIDTH / 2 - 30, HEIGHT - 280))


def display_opponent_bet(env: PokerEnvironment):
    pot_information = BET_FONT.render("Bet: $" + str(env.players[1].current_bet), 1, WHITE)
    WIN.blit(pot_information, (WIDTH / 2 - 30, 190))


def display_sessions_winnings(env: PokerEnvironment):
    winnings = sum(env.players_balance_history[0])
    if winnings < 0:
        text = POT_FONT.render("Session Winnings: -$" + str(-winnings), 1, WHITE)
    else:
        text = POT_FONT.render("Session Winnings: $" + str(winnings), 1, WHITE)
    WIN.blit(text, (70, 40))


def display_user_cards(env: PokerEnvironment):
    WIN.blit(load_card_image(env.players[0].hand[0]), PLAYER_CARD_1)
    WIN.blit(load_card_image(env.players[0].hand[1]), PLAYER_CARD_2)


def display_opponent_cards(env: PokerEnvironment):
    WIN.blit(CARD_BACK, OPPONENT_CARD_1)
    WIN.blit(CARD_BACK, OPPONENT_CARD_2)


def reveal_opponent_cards(env: PokerEnvironment):
    WIN.blit(load_card_image(env.players[1].hand[0]), OPPONENT_CARD_1)
    WIN.blit(load_card_image(env.players[1].hand[1]), OPPONENT_CARD_2)


def display_community_cards(env: PokerEnvironment):
    # Draw the CARDS
    for idx, card in enumerate(env.community_cards):
        if idx == 0:
            WIN.blit(load_card_image(card), FLOP_1_CARD_POSITION)
        elif idx == 1:
            WIN.blit(load_card_image(card), FLOP_2_CARD_POSITION)
        elif idx == 2:
            WIN.blit(load_card_image(card), FLOP_3_CARD_POSITION)
        elif idx == 3:
            WIN.blit(load_card_image(card), TURN_CARD_POSITION)
        else:
            WIN.blit(load_card_image(card), RIVER_CARD_POSITION)


def display_dealer_button(env: PokerEnvironment):
    if env.dealer_button_position == 0:  # User is the dealer
        WIN.blit(dealer_button, (500, HEIGHT - 200))
    else:  # Opponent is the dealer
        WIN.blit(dealer_button, (515, 120))


def draw_window(env: PokerEnvironment, god_mode=False, user_input=False):

    WIN.blit(POKER_BACKGROUND, (0, 0))

    if env.showdown and env.end_of_round():  # Reveal opponent's cards at showdown
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

    # Display Player Balance Information
    display_user_balance(env)
    display_opponent_balance(env)

    # Display Session Winnings
    display_sessions_winnings(env)

    # if env.showdown and env.end_of_round(): # Show who won
    if env.end_of_round():
        winning_players = env.get_winning_players_idx()
        if len(winning_players) == 2:  # Split the pot
            text = BET_FONT.render("This is a tie", 1, WHITE)
        elif winning_players[0] == 0:
            text = BET_FONT.render("You won!", 1, WHITE)
        else:
            text = BET_FONT.render("You lost.", 1, WHITE)

        WIN.blit(text, (250, 350))

    # Pressable Buttons for Check / Fold / Raise. Only display buttons if it is your turn
    warning_text_rendered = BET_FONT.render(warning_text, 1, RED)
    WIN.blit(warning_text_rendered, (WIDTH - 250, HEIGHT - 120))

    if user_input:
        if env.position_in_play == 0 or env.play_as_AI:
            # AAfilledRoundedRect(WIN, RED, pygame.Rect(392,400, 120,50), radius=0.4)
            AAfilledRoundedRect(WIN, RED, check_rect, radius=0.4)
            AAfilledRoundedRect(WIN, RED, custom_rect, radius=0.4)
            AAfilledRoundedRect(WIN, WHITE, input_box, radius=0.4)

            if "f" in env.infoset.actions():
                AAfilledRoundedRect(WIN, RED, fold_rect, radius=0.4)
                fold_bet = BET_BUTTON_FONT.render("Fold", 1, WHITE)
                WIN.blit(fold_bet, (fold_rect.x + 15, fold_rect.y + 7))

            if "k" in env.infoset.actions():
                check_bet = BET_BUTTON_FONT.render("Check", 1, WHITE)
                WIN.blit(check_bet, (check_rect.x + 15, check_rect.y + 7))
            else:  # TODO: Min bet size is not 0 when you are the small blind, so it should be call, not check right.
                # I forgot how the logic is handled for the preflop betting sizes
                call_bet = BET_BUTTON_FONT.render("Call", 1, WHITE)
                WIN.blit(call_bet, (check_rect.x + 28, check_rect.y + 7))

            # TODO: Handle edges cases where these buttons are impossible, in which case you need to grey it out
            custom_bet = BET_BUTTON_FONT.render("Bet", 1, WHITE)
            WIN.blit(custom_bet, (custom_rect.x + 15, custom_rect.y + 7))
            custom_input_bet_text = BET_BUTTON_FONT.render(input_bet_text, 1, BLACK)
            WIN.blit(custom_input_bet_text, (input_box.x + 7, input_box.y + 7))

    if cursor_counter < 15 and active:
        pygame.draw.rect(
            WIN, (0, 0, 0), (WIDTH - 210 + 13 * len(input_bet_text), HEIGHT - 70, 1, 20), 1
        )

    pygame.display.update()


def main():
    score = [0, 0]  # [PLAYER_SCORE, AI_SCORE]
    # Load the nodeMap
    parser = argparse.ArgumentParser(description="Play Hold'Em Poker against the best AI possible.")
    parser.add_argument(
        "-p",
        "--play",
        action="store_true",
        dest="user_input",
        default=True,
        help="Manually play against the AI through a PyGame interface.",
    )
    parser.add_argument(
        "-r",
        "--replay",
        action="store_true",
        dest="replay",
        default=False,
        help="replay a history of games",
    )
    parser.add_argument(
        "-g",
        "--god",
        action="store_true",
        dest="god_mode",
        default=False,
        help="God mode (see the opponent's cards)",
    )

    args = parser.parse_args()
    user_input = args.user_input
    replay = args.replay
    god_mode = args.god_mode

    if replay:  # TODO: Load a history of games, and visualize those
        history = joblib.load("HoldemTrainingHistory.joblib")
        game = 0
        game_i = 0


    env: PokerEnvironment = PokerEnvironment()
    # if user_input or replay:
    #     env.add_player()  # You / replay
    # else:
    #     env.add_AI_player()  # Simulation player

    # if replay:
    #     env.add_player()  # Player since we want everything to be entered manually
    # else:
    #     env.add_AI_player()  # Opponent
    # play as the AI
    env.add_AI_player()
    env.add_player() # play as the opponent too

    clock = pygame.time.Clock()
    run = True

    def place_custom_bet():
        global input_bet_text, warning_text
        if input_bet_text != "":
            bet = "b" + input_bet_text
            print(bet)
            if bet in env.history.actions():
                env.handle_game_stage(bet)
                input_bet_text = ""
                warning_text = ""
            else:
                warning_text = "Invalid bet size"

    while run:
        global input_bet_text, active, cursor_counter, warning_text
        cursor_counter = (cursor_counter + 1) % 30

        if user_input or replay:  # If you want to render PyGame
            clock.tick(FPS)

        handler_called = False

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
            # elif event.type == pygame.VIDEORESIZE: # For resizing of the window
            # 	global WIN
            # 	WIN = pygame.display.set_mode((event.w, event.h), pygame.RESIZABLE)

            # Check if the buttons are clicked, only process if it is our turn
            if user_input:
                if event.type == pygame.MOUSEBUTTONDOWN and env.position_in_play == 0:
                    for i in range(len(buttons)):  # Check for willision with the three buttons
                        if buttons[i].collidepoint(pygame.mouse.get_pos()):
                            warning_text = ""
                            # TODO: Change this for no-limit version
                            if i == 0:
                                env.handle_game_stage("f")  # Fold
                            elif i == 1:
                                if "k" in env.history.actions():
                                    env.handle_game_stage("k")  # Check
                                else:
                                    env.handle_game_stage("c")  # Call
                            elif i == 2:
                                place_custom_bet()

                            handler_called = True
                            break
                if event.type == pygame.MOUSEBUTTONDOWN:
                    # If the user clicked on the input_box rect.
                    if input_box.collidepoint(event.pos):
                        # Toggle the active variable.
                        active = not active
                    else:
                        active = False
                    # Change the current color of the input box.
                    color = color_active if active else color_inactive
                if event.type == pygame.KEYDOWN:
                    if active:
                        if event.key == pygame.K_RETURN:
                            place_custom_bet()

                        elif event.key == pygame.K_BACKSPACE:
                            input_bet_text = input_bet_text[:-1]
                        else:
                            input_bet_text += event.unicode

        if not handler_called:
            if replay:
                if game_i == 0:  # New game, update player's hands
                    # TODO: Show the appropriate community cards. Right now it shows the right player cards, but the board is still the old way.
                    # TODO: This is a little buggy right now too. It doesn't show the right cards.
                    env.players[0].hand = [
                        Card(rank_suit=history[game]["player_cards"][0]),
                        Card(rank_suit=history[game]["player_cards"][1]),
                    ]
                    env.players[1].hand = [
                        Card(rank_suit=history[game]["opponent_cards"][0]),
                        Card(rank_suit=history[game]["opponent_cards"][1]),
                    ]

                env.handle_game_stage(history[game]["history"][game_i])
                game_i += 1
                if game_i >= len(history[game]["history"]):  # Move onto the next game
                    print(
                        "Finished game with history: {}. Player: {} Opponent: {} Board: {}".format(
                            history[game]["history"],
                            history[game]["player_cards"],
                            history[game]["opponent_cards"],
                            history[game]["community_cards"],
                        )
                    )
                    game += 1
                    game_i = 0
                    if game == len(history):
                        print("Finished replay of all games")
                        return

            else:
                env.handle_game_stage()

        # At Showdown, reveal opponent's cards and add a delay
        if replay or user_input:
            draw_window(env, god_mode, user_input)

        if user_input and env.end_of_round():
            draw_window(env, god_mode, False)
            time.sleep(2)

    pygame.quit()


if __name__ == "__main__":
    main()
