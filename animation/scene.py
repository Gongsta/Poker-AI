from manim import *
import networkx as nx
"""
Lif of animations I need to make:
- [] Text: Nash Equibrium + Definition
- [] Animation: Rock-Paper-Scissors visualized
- [] Animation: Game tree for chess, animated
- [] Animation: Regret Matching algorithm
- [] Text: Counterfactual Regret Minimization
- [] Animation: actual CFR Algorithm
- [] Animation: Card Abstraction? And Bet Abstraction?
"""

# Some options of background colors
# config.background_color = "#151213" # Dark brown Gray
# config.background_color = "#262627" # Gray
# config.background_color = "#121212" # Dark Gray
# config.background_color = PURE_GREEN
config.background_color = BLACK
Text.set_default(font='Shadows Into Light', color=BLACK)

class NashEquilibriumText(Scene):
    def construct(self):
        nashEquilibrium = Tex('Nash Equilibrium',font_size=100)
        self.play(Write(nashEquilibrium, run_time=1.5))
        d1 = Tex(r'A scenario in game theory in which no')
        d2 = Tex(r'players can improve by deviating from')
        d3 = Tex(r'their strategy.')
        d = VGroup(d1,d2,d3).arrange(direction=DOWN, aligned_edge=LEFT, buff=0.2)
        self.play(AnimationGroup(nashEquilibrium.animate.shift(2 * UP), FadeIn(d, run_time=1.5), lag_ratio=1))
        self.wait(2)
        self.play(FadeOut(*self.mobjects)) # TODO: Maybe link this together with the next animation


class bits(Scene):
    def construct(self):
        MONOLISA_FONT = 'MonoLisa'
        cardsText = Text('52 Cards', font=MONOLISA_FONT)
        bits = Text(r'0000000000000000000000000000000000000000000000000', font=MONOLISA_FONT, font_size=28)
        self.play(ReplacementTransform(cardsText, bits))
        bits_with_cards = Text(r'0010001000000000000000000000000000000000000000100', font=MONOLISA_FONT, font_size=28)
        self.play(ReplacementTransform(bits, bits_with_cards))
        clubs_02 = ImageMobject('../assets/cards/card_clubs_02.png')
        self.play(FadeIn(clubs_02))
        # self.play(ReplacementTransform(bits_with_cards, clubs_02))


def create_mobject(choice):
    return ImageMobject(f'assets/{choice}.png').scale(0.5)


class RPS(Scene):
    def construct(self):
        background = ImageMobject('assets/background.png').scale(0.4)
        self.add(background)

        rock = create_mobject('rock').shift(4.5*LEFT)
        paper = create_mobject('paper')
        scissors = create_mobject('scissors').shift(4.5*RIGHT)
        
        self.play(FadeIn(rock, scale=0.5), FadeIn(paper, scale=0.5), FadeIn(scissors, scale=0.5))
        # self.play(ReplacementTransform(rText, rock), ReplacementTransform(pText, paper), ReplacementTransform(sText, scissors))
        
        self.wait(1)
        
        opponentText = Paragraph("Opponent\nChoice", font_size=40, alignment='center').shift(2.7*UP)
        arrow = Arrow(ORIGIN, DOWN, buff=0.1, stroke_width=3, color=BLACK).next_to(opponentText, DOWN)
        self.play(Write(opponentText), rock.animate.shift(0.5*DOWN), paper.animate.shift(0.5*DOWN), scissors.animate.shift(0.5*DOWN))
        # oval = Ellipse(width=4, height=8, color=BLACK, fill_opacity=0)
        highlight = ImageMobject('assets/emphasize.png').scale(0.6).stretch(0.92, 1).shift(0.2*UP).stretch(0.9, 0).scale(0)
        self.play(AnimationGroup(Create(arrow), FadeIn(highlight), lag_ratio=0.5))
        group = Group(opponentText, arrow, highlight)
        self.play(group.animate.shift(4.5*RIGHT))
        self.wait(0.3)
        rock2 = ImageMobject('assets/question.png').scale(0.5).shift(4.5*LEFT).shift(0.5*DOWN)
        paper2 = ImageMobject('assets/question.png').scale(0.5).shift(0.5*DOWN)
        scissors2 = ImageMobject('assets/question.png').scale(0.5).shift(4.5*RIGHT).shift(0.5*DOWN)
        self.play(FadeIn(rock2), FadeIn(paper2), FadeIn(scissors2), group.animate.shift(9*LEFT), FadeOut(scissors), FadeOut(paper), FadeOut(rock))
        self.wait(0.3)
        self.play(group.animate.shift(4.5*RIGHT))
        self.wait(0.3)
        self.play(group.animate.shift(4.5*LEFT))
        self.wait(0.3)
        self.play(group.animate.shift(9*RIGHT))
        self.wait(0.8)

        to_remove = []
        for obj in self.mobjects:
            if obj != background:
                to_remove.append(obj)
        self.play(FadeOut(*to_remove))

        
class RPS2(Scene):
    def construct(self):
        background = ImageMobject('assets/background.png').scale(0.4)
        self.add(background)

        rock = create_mobject('rock').shift(4*LEFT + 2*UP)
        paper = create_mobject('paper').shift(2*UP)
        scissors = create_mobject('scissors').shift(4*RIGHT + 2*UP)
        
        self.play(AnimationGroup(FadeIn(rock, scale=0.4), FadeIn(paper, scale=0.4), FadeIn(scissors, scale=0.4), lag_ratio=0.7))
        self.wait(1)

        rock2 = rock.copy().next_to(scissors, 5*DOWN)
        paper2 = paper.copy().next_to(rock, 3*DOWN)
        scissors2 = scissors.copy().next_to(paper, 3*DOWN)
        
        x1 = [rock.get_x(), 1, 0]
        x2 = [paper.get_x(), 1, 0]
        x3 = [scissors.get_x(), 1, 0]
        line1 = Line(x1, x1 + DOWN, color=BLACK)
        line2 = Line(x2, x2 + DOWN, color=BLACK)
        line3 = Line(x3, x3 + DOWN, color=BLACK)
        
        line1.add_updater(lambda m: m.put_start_and_end_on([min(rock.get_x(), -0.5), 1, 0], x1 + DOWN))
        line2.add_updater(lambda m: m.put_start_and_end_on([paper.get_x(), 1, 0], x2 + DOWN))
        line3.add_updater(lambda m: m.put_start_and_end_on([max(scissors.get_x(), 0.5), 1, 0], x3 + DOWN))
        
        self.play(AnimationGroup(FadeIn(paper2, scale=0.4), FadeIn(scissors2, scale=0.4), FadeIn(rock2, scale=0.4), lag_ratio=0.7), AnimationGroup(Create(line1), Create(line2), Create(line3), lag_ratio=0.7))
        self.wait(1)
        
        q = ImageMobject('assets/question.png').scale(0.5).shift(2*UP)
        self.play(AnimationGroup(AnimationGroup(rock.animate.shift(4*RIGHT), scissors.animate.shift(4*LEFT)), FadeIn(q), lag_ratio=0.3))
        self.play(FadeOut(rock), FadeOut(scissors), FadeOut(paper))
        self.wait(2)
        self.play(AnimationGroup(AnimationGroup(FadeOut(line1), FadeOut(line2), FadeOut(line3), FadeOut(q)), AnimationGroup(rock2.animate.shift(2.5*UP), paper2.animate.shift(2.5*UP), scissors2.animate.shift(2.5*UP)), lag_ratio=0.3))
        
        one_third_text = Text("1/3", font_size=60).next_to(paper2, 2*DOWN)
        one_third_text2 = Text("1/3", font_size=60).next_to(rock2, 2*DOWN)
        one_third_text3 = Text("1/3", font_size=60).next_to(scissors2, 2*DOWN)
        
        self.play(Write(one_third_text), Write(one_third_text2), Write(one_third_text3))
        self.wait(2)
        self.play(FadeOut(one_third_text), FadeOut(one_third_text2), FadeOut(one_third_text3), FadeOut(rock2), FadeOut(paper2), FadeOut(scissors2))


# config.background_color = GRAY_BROWN
# Text.set_default(font='Shadows Into Light', color=WHITE)

class RPSSim(Scene):
    """
    When you play randomly
    """
    def construct(self):
        # TODO: Same code as below
        return

class RPSSimRock(Scene):
    """
    When you play rock all the time, you opponent catches on after 10 iterations
    """

    def construct(self):
        plane = NumberPlane(
            x_range = (0, 1),
            y_range = (0, 1, 0.5),
            x_length=4,
            y_length=4,
            axis_config={"include_numbers": True},
            y_axis_config={"label_direction": LEFT},
        )
        # plane.center()
        plane.shift(4*RIGHT)
        x_values = [0]
        y_values = [0]
        line_graph = plane.plot_line_graph(
            x_values = x_values,
            y_values = y_values,
            line_color=GOLD_E,
            vertex_dot_style=dict(stroke_width=3),
            stroke_width = 4,
        )
        self.add(plane, line_graph)
        
        
        your_score_tracker = ValueTracker(0)
        opponent_score_tracker = ValueTracker(0)
        you = Text('You').shift(5*LEFT + 2*UP)
        opp= Text('Opponent').shift(LEFT + 2*UP)
        your_score_placeholder = Text('0', font_size=80).next_to(you, DOWN)
        opponent_score_placeholder = Text('0', font_size=80).next_to(opp, DOWN)
        
        self.add(you, opp, your_score_placeholder, opponent_score_placeholder)
        
        self.add(Text('Win Rate over Time', font_size=30).next_to(plane, UP))
        self.add(Text('Time', font_size=20).next_to(plane, DOWN).shift(2*RIGHT))
        self.add(Text('Win Rate', font_size=20).next_to(plane, LEFT).shift(2*UP))

        for i in range(1,20):
            group = ['rock', 'paper', 'scissors']
            # choice_player = np.random.randint(0,3)
            choice_player = 0
            if i < 5:
                choice_opponent = np.random.randint(0,3)
            else:
                choice_opponent = 1
            
            if choice_player == choice_opponent:
                text = Text("Tie.", font_size=50)
            elif (choice_player - choice_opponent) % 3 == 1:
                text = Text("Win!", font_size=50)
                your_score_tracker += 1
            else:
                text = Text("Loss :(", font_size=50)
                opponent_score_tracker += 1
            
            x_values.append(i)
            y_values.append(your_score_tracker.get_value() / (your_score_tracker.get_value() + opponent_score_tracker.get_value()))

            player = create_mobject(group[choice_player]).move_to(5*LEFT + DOWN)
            opponent = create_mobject(group[choice_opponent]).move_to(LEFT + DOWN)

            your_score_updated = Text(str(int(your_score_tracker.get_value())), font_size=80).next_to(you, DOWN)
            opponent_score_updated = Text(str(int(opponent_score_tracker.get_value())), font_size=80).next_to(opp, DOWN)

            text.shift(DOWN + 3*LEFT)
            self.play(AnimationGroup(FadeIn(player), FadeIn(opponent),
                        ), run_time=0.1)
            self.add(text)
            self.play(AnimationGroup(your_score_placeholder.animate.become(your_score_updated), opponent_score_placeholder.animate.become(opponent_score_updated), run_time=0.2))
            # self.wait(0.2)
            
            new_plane = NumberPlane(
                x_range = (0, i),
                y_range = (0, 1, round(max(1/(i+1), 0.2), 1)),
                x_length=4,
                y_length=4,
                axis_config={"include_numbers": True},
                y_axis_config={"label_direction": LEFT},
            )
            new_plane.shift(4*RIGHT)

            if i <=6:
                run_time = 0.2
            else:
                run_time = 0.02
            self.play(line_graph.animate.become(new_plane.plot_line_graph(
                x_values = x_values,
                y_values = y_values,
                line_color=GOLD_E,
                vertex_dot_style=dict(stroke_width=3),
                stroke_width = 4,
            )), plane.animate.become(new_plane), run_time=run_time)

            self.play(AnimationGroup(FadeOut(player), FadeOut(opponent), run_time=0.1))
            self.remove(text)
            


            




class RPSold(Scene):
    """
    Not using this because it seems too complicated for no reason. Remember that simplicity is key.
    
    But actually, game trees are useful. but this is too complicated redo it.
    
    Maybe use this to say that if you knew what you opponent was going to play, then 
    """
    def construct(self):
        rText = Tex("Rock", font_size=100).shift(3.5*LEFT)
        pText = Tex("Paper", font_size=100).shift(0.1 * DOWN, 0.5 * LEFT)
        sText = Tex("Scissors", font_size=100).shift(3*RIGHT)

        youText = Tex("You", font_size=50).shift(2*LEFT + UP)
        vs = Tex("vs.", font_size=20)
        opponentText = Tex("Opponent", font_size=50).shift(2*RIGHT + UP)

        self.play(Write(rText), Write(pText), Write(sText))
        self.play(Transform(rText, youText), Transform(pText, vs), ReplacementTransform(sText, opponentText))
        
        rock = create_mobject('rock').shift(2 * LEFT)
        self.play(FadeIn(rock))
        q = Text("?").scale(2).shift(2 * RIGHT)
        self.play(FadeIn(q))
        
        startPos = 2*UP
        blueStart = Circle(0.3, color=RED).shift(startPos)
        player_text = Tex("Opponent", font_size=36).shift(startPos + 2*RIGHT)
        self.play(AnimationGroup(Transform(opponentText, blueStart), Write(player_text), lag_ratio=0.6))
        # This is a "hack" to allow me to create duplicates to transform one node into three nodes
        blueStart1 = Circle(0.3, color=RED).move_to(blueStart.get_center())
        blueStart2 = Circle(0.3, color=RED).move_to(blueStart.get_center())
        blueStart3 = Circle(0.3, color=RED).move_to(blueStart.get_center())
        
        redRock = Circle(0.3, color=BLUE).move_to(blueStart.get_center()).shift(3*LEFT + 2 * DOWN)
        redPaper = Circle(0.3, color=BLUE).move_to(blueStart.get_center()).shift(2 * DOWN)
        redScissors = Circle(0.3, color=BLUE).move_to(blueStart.get_center()).shift(3*RIGHT + 2 * DOWN)
        
        redRPS = [redRock, redPaper, redScissors]
        
        # Edges between 1st layer and 2nd layer
        line1 = Line(LEFT, LEFT) # intialize empty line
        line2 = Line(LEFT, LEFT) # intialize empty line
        line3 = Line(LEFT, LEFT) # intialize empty line
        line1.add_updater(lambda z: z.become(Line(normalize(blueStart1.get_center() -  blueStart.get_center()) * 0.3 + blueStart.get_center(), normalize(-blueStart1.get_center() + blueStart.get_center()) * 0.3 + blueStart1.get_center())) if blueStart.point_at_angle(225*DEGREES)[0] > blueStart1.point_at_angle(45*DEGREES)[0] else None)
        line2.add_updater(lambda z: z.become(Line(blueStart.point_at_angle(270*DEGREES), blueStart2.point_at_angle(90*DEGREES))) if blueStart.point_at_angle(270*DEGREES)[1] > blueStart2.point_at_angle(90*DEGREES)[1] else None)
        line3.add_updater(lambda z: z.become(Line(normalize(blueStart3.get_center() -  blueStart.get_center()) * 0.3 + blueStart.get_center(), normalize(-blueStart3.get_center() + blueStart.get_center()) * 0.3 + blueStart3.get_center())) if blueStart.point_at_angle(315*DEGREES)[0] < blueStart3.point_at_angle(135*DEGREES)[0] else None)
        self.add(line1, line2, line3)

        # self.play(AnimationGroup(Transform(blueStart1, redRPS[0]), Transform(blueStart2, redRPS[1]), Transform(blueStart3, redRPS[2])))
        # lag_ratio = 0.5

        opponent_text = Tex("You", font_size=36).move_to(redScissors.get_center()).shift(1.5*RIGHT)
        rockVec = redRock.get_center() -  blueStart.get_center()
        scissorsVec = redScissors.get_center() -  blueStart.get_center()
        rockText = Tex("Rock", font_size=28).rotate(np.arctan(rockVec[1] /rockVec[0])).move_to(line1.point_from_proportion(0.6) + 0.3 *UP)
        paperText = Tex("Paper", font_size=28).rotate(-PI/2).move_to(line2.point_from_proportion(0.5) + 0.23 *RIGHT)
        scissorsText = Tex("Scissors", font_size=28).rotate(np.arctan(scissorsVec[1] /scissorsVec[0])).move_to(line3.point_from_proportion(0.6) + 0.3 *UP)
        rockText.add_updater(lambda z: z.move_to(line1.point_from_proportion(0.6) + 0.3 *UP))
        paperText.add_updater(lambda z: z.move_to(line2.point_from_proportion(0.5) + 0.23 *RIGHT))
        scissorsText.add_updater(lambda z: z.move_to(line3.point_from_proportion(0.6) + 0.3 *UP))
        # self.play(Write(rockText), Write(paperText), Write(scissorsText),Write(opponent_text))
        self.play(AnimationGroup(AnimationGroup(Transform(blueStart1, redRPS[0]), Transform(blueStart2, redRPS[1]), Transform(blueStart3, redRPS[2])), AnimationGroup(Write(rockText), Write(paperText), Write(scissorsText),Write(opponent_text)), lag_ratio=0.7))
        opponent_text.add_updater(lambda z: z.move_to(redScissors.get_center()).shift(1.5*RIGHT))
        
        redRPSDuplicates = []
        for val in redRPS:
            redRPSDuplicates.append([Circle(0.3, color=BLUE).move_to(val.get_center()) for _ in range(3)])


        flatRedRPSDuplicates = [item for ll in redRPSDuplicates for item in ll]
        self.add(*flatRedRPSDuplicates)
        redRPSTransforms = []
        for i in range(3): # i == 0 -> Rock, i == 1 -> Paper, i == 2 -> Scissors
            redRPSTransforms.append([])
            for j in range(3):
                if j == 0:
                    redRPSTransforms[i].append(Transform(redRPSDuplicates[i][0], Circle(0.3, color=GREY).move_to(redRPS[i].get_center()).shift(LEFT+2*DOWN)))
                elif j == 1:
                    redRPSTransforms[i].append(Transform(redRPSDuplicates[i][1], Circle(0.3, color=GREY).move_to(redRPS[i].get_center()).shift(2*DOWN)))
                else:
                    redRPSTransforms[i].append(Transform(redRPSDuplicates[i][2], Circle(0.3, color=GREY).move_to(redRPS[i].get_center()).shift(RIGHT + 2*DOWN)))
        
        
        lines = []
        lines.append([Line(LEFT, LEFT) for _ in range(3)])
        lines.append([Line(LEFT, LEFT) for _ in range(3)])
        lines.append([Line(LEFT, LEFT) for _ in range(3)])
        lines[0][0].add_updater(lambda z: z.become(Line(redRPS[0].point_at_angle(225*DEGREES), redRPSDuplicates[0][0].point_at_angle(90*DEGREES))) if redRPS[0].point_at_angle(225*DEGREES)[0] > redRPSDuplicates[0][0].point_at_angle(90*DEGREES)[0] else None)
        lines[0][1].add_updater(lambda z: z.become(Line(redRPS[0].point_at_angle(270*DEGREES), redRPSDuplicates[0][1].point_at_angle(90*DEGREES))) if redRPS[0].point_at_angle(270*DEGREES)[1] > redRPSDuplicates[0][1].point_at_angle(90*DEGREES)[1] else None)
        lines[0][2].add_updater(lambda z: z.become(Line(redRPS[0].point_at_angle(315*DEGREES), redRPSDuplicates[0][2].point_at_angle(90*DEGREES))) if redRPS[0].point_at_angle(315*DEGREES)[0] < redRPSDuplicates[0][2].point_at_angle(90*DEGREES)[0] else None)
        lines[1][0].add_updater(lambda z: z.become(Line(redRPS[1].point_at_angle(225*DEGREES), redRPSDuplicates[1][0].point_at_angle(90*DEGREES))) if redRPS[1].point_at_angle(225*DEGREES)[0] > redRPSDuplicates[1][0].point_at_angle(90*DEGREES)[0] else None)
        lines[1][1].add_updater(lambda z: z.become(Line(redRPS[1].point_at_angle(270*DEGREES), redRPSDuplicates[1][1].point_at_angle(90*DEGREES))) if redRPS[1].point_at_angle(270*DEGREES)[1] > redRPSDuplicates[1][1].point_at_angle(90*DEGREES)[1] else None)
        lines[1][2].add_updater(lambda z: z.become(Line(redRPS[1].point_at_angle(315*DEGREES), redRPSDuplicates[1][2].point_at_angle(90*DEGREES))) if redRPS[1].point_at_angle(315*DEGREES)[0] < redRPSDuplicates[1][2].point_at_angle(90*DEGREES)[0] else None)
        lines[2][0].add_updater(lambda z: z.become(Line(redRPS[2].point_at_angle(225*DEGREES), redRPSDuplicates[2][0].point_at_angle(90*DEGREES))) if redRPS[2].point_at_angle(225*DEGREES)[0] > redRPSDuplicates[2][0].point_at_angle(90*DEGREES)[0] else None)
        lines[2][1].add_updater(lambda z: z.become(Line(redRPS[2].point_at_angle(270*DEGREES), redRPSDuplicates[2][1].point_at_angle(90*DEGREES))) if redRPS[2].point_at_angle(270*DEGREES)[1] > redRPSDuplicates[2][1].point_at_angle(90*DEGREES)[1] else None)
        lines[2][2].add_updater(lambda z: z.become(Line(redRPS[2].point_at_angle(315*DEGREES), redRPSDuplicates[2][2].point_at_angle(90*DEGREES))) if redRPS[2].point_at_angle(315*DEGREES)[0] < redRPSDuplicates[2][2].point_at_angle(90*DEGREES)[0] else None)

        
        flatLines = [item for line in lines for item in line]
        self.add(*flatLines)
        flatRedRPSTransforms = [item for sublist in redRPSTransforms for item in sublist]
        end_of_game_text = Tex("End of Game", font_size=28).move_to(redRPS[2].get_center()).shift(2.5*RIGHT + 2*DOWN)

        rockVec = LEFT + 2*DOWN
        scissorsVec = RIGHT + 2 * DOWN
        rockText1 = Tex("Rock", font_size=20).rotate(np.arctan(rockVec[1] /rockVec[0])).move_to(lines[0][0].point_from_proportion(0.6) + 0.3 *UP)
        paperText1 = Tex("Paper", font_size=20).rotate(-PI/2).move_to(lines[0][1].point_from_proportion(0.5) + 0.15 *RIGHT)
        scissorsText1 = Tex("Scissors", font_size=20).rotate(np.arctan(scissorsVec[1] /scissorsVec[0])).move_to(lines[0][2].point_from_proportion(0.6) + 0.3 *UP)
        rockText2 = Tex("Rock", font_size=20).rotate(np.arctan(rockVec[1] /rockVec[0])).move_to(lines[1][0].point_from_proportion(0.6) + 0.3 *UP)
        paperText2 = Tex("Paper", font_size=20).rotate(-PI/2).move_to(lines[1][1].point_from_proportion(0.5) + 0.15 *RIGHT)
        scissorsText2 = Tex("Scissors", font_size=20).rotate(np.arctan(scissorsVec[1] /scissorsVec[0])).move_to(lines[1][2].point_from_proportion(0.6) + 0.3 *UP)
        rockText3 = Tex("Rock", font_size=20).rotate(np.arctan(rockVec[1] /rockVec[0])).move_to(lines[2][0].point_from_proportion(0.6) + 0.3 *UP)
        paperText3 = Tex("Paper", font_size=20).rotate(-PI/2).move_to(lines[2][1].point_from_proportion(0.5) + 0.15 *RIGHT)
        scissorsText3 = Tex("Scissors", font_size=20).rotate(np.arctan(scissorsVec[1] /scissorsVec[0])).move_to(lines[2][2].point_from_proportion(0.6) + 0.3 *UP)
        rockText1.add_updater(lambda z: z.move_to(lines[0][0].point_from_proportion(0.7) + 0.3 *UP))
        paperText1.add_updater(lambda z: z.move_to(lines[0][1].point_from_proportion(0.5) + 0.15 *RIGHT))
        scissorsText1.add_updater(lambda z: z.move_to(lines[0][2].point_from_proportion(0.7) + 0.3 *UP))
        rockText2.add_updater(lambda z: z.move_to(lines[1][0].point_from_proportion(0.7) + 0.3 *UP))
        paperText2.add_updater(lambda z: z.move_to(lines[1][1].point_from_proportion(0.5) + 0.15 *RIGHT))
        scissorsText2.add_updater(lambda z: z.move_to(lines[1][2].point_from_proportion(0.7) + 0.3 *UP))
        rockText3.add_updater(lambda z: z.move_to(lines[2][0].point_from_proportion(0.7) + 0.3 *UP))
        paperText3.add_updater(lambda z: z.move_to(lines[2][1].point_from_proportion(0.5) + 0.15 *RIGHT))
        scissorsText3.add_updater(lambda z: z.move_to(lines[2][2].point_from_proportion(0.7) + 0.3 *UP))

        end_of_game_text.add_updater(lambda z: z.move_to(redRPS[2].get_center()).shift(2.5*RIGHT + 2 * DOWN))
        self.play(AnimationGroup(AnimationGroup(*flatRedRPSTransforms), AnimationGroup(Write(end_of_game_text), 
        Write(rockText1), Write(rockText2), Write(rockText3), Write(paperText1), Write(paperText2), Write(paperText3), Write(scissorsText1), Write(scissorsText2), Write(scissorsText3)
        ), lag_ratio=0.7))
        self.wait(1)
        # merge the nodes to show that we don't know
        """
        Steps:
        1. Replace text with question mark
        2. Move red nodes into one
        """
        # update the updaters so the lines can overlap
        line1.clear_updaters()
        line3.clear_updaters()
        line1.add_updater(lambda z: z.become(Line(normalize(blueStart1.get_center() -  blueStart.get_center()) * 0.3 + blueStart.get_center(), normalize(-blueStart1.get_center() + blueStart.get_center()) * 0.3 + blueStart1.get_center())))
        line3.add_updater(lambda z: z.become(Line(normalize(blueStart3.get_center() -  blueStart.get_center()) * 0.3 + blueStart.get_center(), normalize(-blueStart3.get_center() + blueStart.get_center()) * 0.3 + blueStart3.get_center())))
        
        # Remove the characters
        self.play(FadeOut(rockText, paperText, scissorsText),
            VGroup(redRPSDuplicates[0][0], redRPSDuplicates[0][1], redRPSDuplicates[0][2], redRPS[0],
        ).animate.shift(redPaper.get_center() - redRPS[0].get_center()),
        VGroup(redRPSDuplicates[2][0], redRPSDuplicates[2][1], redRPSDuplicates[2][2], redRPS[2],
        ).animate.shift(redPaper.get_center() - redRPS[2].get_center()),
        blueStart1.animate.shift(redPaper.get_center() - blueStart1.get_center()),
        blueStart3.animate.shift(redPaper.get_center() - blueStart3.get_center()),
        )
        questionMark = Tex('?').next_to(line2, RIGHT)
        self.play(Write(questionMark))
        
        # self.play(redRock.animate.move_to(redPaper), redScissors.animate.move_to(redPaper))

        # self.play(Create(bluePaper, blueRock, blueScissors))

class badRPS(Scene):
    # Idea: Start from drawing out the full game tree, and collapse opponent nodes into one
    def construct(self):
        G = nx.Graph()
        G.add_node("ROOT")

        choices = ["Rock", "Paper", "Scissors"]
        players = ["Opponent", "You"]
        for player in ["Opponent", "You"]:
            for choice in choices:
                G.add_node(f'{player}_{choice}')
                
                if player == "Opponent":
                    G.add_edge("ROOT", f'{player}_{choice}')
                else:
                    for opp_choice in choices:
                        G.add_edge(f'Opponent_{opp_choice}', f'{player}_{choice}')

        
        gg = Graph(G.nodes, G.edges, root_vertex="ROOT", layout="circular", vertex_config={'radius': 0.2}, labels=True)
        self.play(Create(gg))
        # self.play(gg.animate.change_layout("tree", root_vertex="ROOT"))
        # self.wait()

class CFRText(Scene):
    def construct(self):
        text = Text('Counterfactual Regret').scale(2)
        text2 = Text('Minimization (CFR)').scale(2).next_to(text, DOWN)
        self.play(Write(text))
        self.play(Write(text2))
        self.wait(2)
        self.play(FadeOut(*self.mobjects))

class hist(Scene):
    """
    Animation for the Histograms for the card abstraction
    
    Idea:
    Start with the cards on each node.
    Then, you use a histogram representation.
    


    
    """
    def construct(self):
        # distributions = np.random.random((10,5))
        values = [0.3,0.5,0.2]
        chart = BarChart(
            values
        )
        self.play(Create(chart))

        values2 = [0.5,0.2,0.4]
        chart2 = BarChart(
            values
        )
        self.play(Transform(chart, chart2))


class bet(Scene):
    """
    Idea: 
    Have a continuous line, and then bucket these bets into discrete values.
    
    """
    def construct(self):
        text_102 = Tex('102\$', font_size=64)
        text_100 = Tex('100\$', font_size=64)
        self.play(Transform(text_102, text_100))
        # number_line = NumberLine(
        #     x_range=[0, 100, 10],
        #     length=10,
        # )
        # check_text = Tex('Check').next_to(number_line.n2p(0), DOWN)
        # all_in_text = Tex('All-In').next_to(number_line.n2p(100), DOWN)
        # self.play(AnimationGroup(Create(number_line), Write(check_text), Write(all_in_text), lag_ratio=0.3))
        # self.wait(2)
        
        start_node = Circle(0.3, color=BLUE).shift(2*UP)
        target_nodes = VGroup(*[Circle(0.05, color=RED) for _ in range(60)])
        target_nodes.arrange(RIGHT, buff=0.1).shift(2*DOWN)
        lines = []
        for i in range(60):
            unit_v = normalize(target_nodes[i].get_center() - start_node.get_center())
            lines.append(Line(start_node.get_center() + 0.3 * unit_v, target_nodes[i].get_center() - 0.05 * unit_v, color=GRAY))
        Line(start_node)
        self.play(Create(start_node))
        self.play(Create(target_nodes), Create(VGroup(*lines)))
        self.wait(2)

        target_nodes_2 = VGroup(*[Circle(0.2, color=RED) for _ in range(10)])
        target_nodes_2.arrange(RIGHT, buff=0.2).shift(2*DOWN)
        lines2 = []
        for i in range(10):
            unit_v = normalize(target_nodes_2[i].get_center() - start_node.get_center())
            lines2.append(Line(start_node.get_center() + 0.3 * unit_v, target_nodes_2[i].get_center() - 0.2 * unit_v, color=GRAY))
        
        self.play(Transform(VGroup(*target_nodes, *lines), VGroup(*target_nodes_2, *lines2)))


class valueTemplate(Scene):
    def construct(self):
        number_line = NumberLine()
        pointer = Vector(DOWN).shift(UP)
        label = Tex("x")
        label.add_updater(lambda m: m.next_to(pointer, UP))
        
        tracker = ValueTracker(0)
        pointer.add_updater(lambda m: m.next_to(
            number_line.n2p(tracker.get_value()), UP)
        )
        self.add(number_line, pointer, label)
        tracker += 1.5
        self.wait(1)
        tracker -= 4
        self.wait(0.5)
        self.play(tracker.animate.set_value(5))
    

# class Histogram(Scene):
#     """
#     Create a template for this histogram, which you will be able to recycle in the future:
#     """
#     # Add Image
#     corona= ImageMobject("assets/img/covid_19.png")
#     corona.scale(1.2)
#     corona.to_edge(RIGHT, buff=1)

#     self.add(corona)
        



class GraphExample(Scene):
    def construct(self):
        ax = Axes(x_range=[0,5,1], y_range=[0,3,1])