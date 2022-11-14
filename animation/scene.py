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
