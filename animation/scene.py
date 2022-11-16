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

# config.background_color = PURE_GREEN

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

class RPS(Scene):
    def construct(self):
        
        startPos = 2*UP
        blueStart = Circle(0.3, color=BLUE).shift(startPos)
        player_text = Tex("You", font_size=36).shift(startPos + 1.5*RIGHT)
        self.play(AnimationGroup(Create(blueStart), Write(player_text), lag_ratio=0.4))
        blueStart1 = Circle(0.3, color=BLUE).move_to(blueStart.get_center())
        blueStart2 = Circle(0.3, color=BLUE).move_to(blueStart.get_center())
        blueStart3 = Circle(0.3, color=BLUE).move_to(blueStart.get_center())
        
        
        redRock = Circle(0.3, color=RED).move_to(blueStart.get_center()).shift(3*LEFT + 2 * DOWN)
        redPaper = Circle(0.3, color=RED).move_to(blueStart.get_center()).shift(2 * DOWN)
        redScissors = Circle(0.3, color=RED).move_to(blueStart.get_center()).shift(3*RIGHT + 2 * DOWN)
        
        redRPS = [redRock, redPaper, redScissors]
        
        # Edges between 1st layer and 2nd layer
        line1 = Line(LEFT, LEFT) # intialize empty line
        line2 = Line(LEFT, LEFT) # intialize empty line
        line3 = Line(LEFT, LEFT) # intialize empty line
        line1.add_updater(lambda z: z.become(Line(blueStart.point_at_angle(225*DEGREES), blueStart1.point_at_angle(45*DEGREES))) if blueStart.point_at_angle(225*DEGREES)[0] > blueStart1.point_at_angle(45*DEGREES)[0] else None)
        line2.add_updater(lambda z: z.become(Line(blueStart.point_at_angle(270*DEGREES), blueStart2.point_at_angle(90*DEGREES))) if blueStart.point_at_angle(270*DEGREES)[1] > blueStart2.point_at_angle(90*DEGREES)[1] else None)
        line3.add_updater(lambda z: z.become(Line(blueStart.point_at_angle(315*DEGREES), blueStart3.point_at_angle(135*DEGREES))) if blueStart.point_at_angle(315*DEGREES)[0] < blueStart3.point_at_angle(135*DEGREES)[0] else None)
        self.add(line1, line2, line3)

        opponent_text = Tex("Opponent", font_size=36).move_to(redScissors.get_center()).shift(2*RIGHT)
        self.play(AnimationGroup(AnimationGroup(Transform(blueStart1, redRPS[0]), Transform(blueStart2, redRPS[1]), Transform(blueStart3, redRPS[2])),
        Write(opponent_text), lag_ratio=0.5)
        )
        
        redRPSDuplicates = []
        for val in redRPS:
            redRPSDuplicates.append([Circle(0.3, color=RED).move_to(val.get_center()) for _ in range(3)])


        flatRedRPSDuplicates = [item for ll in redRPSDuplicates for item in ll]
        self.add(*flatRedRPSDuplicates)
        redRPSTransforms = []
        for i in range(3): # i == 0 -> Rock, i == 1 -> Paper, i == 2 -> Scissors
            redRPSTransforms.append([])
            for j in range(3):
                if j == 0:
                    redRPSTransforms[i].append(Transform(redRPSDuplicates[i][0], Circle(0.3, color=BLUE).move_to(redRPS[i].get_center()).shift(LEFT+2*DOWN)))
                elif j == 1:
                    redRPSTransforms[i].append(Transform(redRPSDuplicates[i][1], Circle(0.3, color=BLUE).move_to(redRPS[i].get_center()).shift(2*DOWN)))
                else:
                    redRPSTransforms[i].append(Transform(redRPSDuplicates[i][2], Circle(0.3, color=BLUE).move_to(redRPS[i].get_center()).shift(RIGHT + 2*DOWN)))
        
        
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
        player_text_2 = Tex("You", font_size=36).move_to(redRPS[i].get_center()).shift(2.5*RIGHT + 2*DOWN)
        self.play(AnimationGroup(AnimationGroup(*flatRedRPSTransforms), Write(player_text_2), lag_ratio=0.5))

        self.wait(2)
        
        
        # merge the nodes to show that we don't know
        """
        Steps:
        1. Replace text with question mark
        2. Move red nodes into one
        """
        self.play(redRock.animate.move_to(redPaper), redScissors.animate.move_to(redPaper))

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


class BetAbstraction(Scene):
    """
    Idea: 
    Have a continuous line, and then bucket these bets into discrete values.
    
    """
    def construct(self):
        tex
        
        



class GraphExample(Scene):
    def construct(self):
        ax = Axes(x_range=[0,5,1], y_range=[0,3,1])