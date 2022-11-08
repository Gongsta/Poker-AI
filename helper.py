from pygame import Rect, Color, Surface, transform, SRCALPHA, draw, BLEND_RGBA_MIN, BLEND_RGBA_MAX
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib
import networkx as nx

"""
Animation Helper Functions
"""
matplotlib.rcParams['axes.spines.right'] = False
matplotlib.rcParams['axes.spines.top'] = False
matplotlib.rcParams['axes.spines.left'] = False
matplotlib.rcParams['axes.spines.bottom'] = False

def animate_rps(history):
	# To visualize Rock Paper Scissors bar charts over time
	fig, ax = plt.subplots()
	def animate(i):
		fig.clear()
		plt.title(f"Iteration = {i}")
		plt.ylabel("Strategy")
		plt.bar(["Rock", "Paper", "Scissors"], history[i])  #, color=['tab:blue', 'tab:orange', 'tab:green'])
		ax.tick_params(bottom="off")
		# plt.axis("off")


	animation = FuncAnimation(fig, animate, len(history), repeat=False, interval=30)
	plt.show()
	# animation.save("animation.gif")
	
def create_RPS_node():
	G = nx.Graph()
	G.add_edge(0, 1)
	G.add_edge(0, 2)
	G.add_edge(0, 3)
	for i in range(1, 4):
		for j in range(i*3+1, i*3 + 4):
			G.add_edge(i, j)
	subax1 = plt.subplot()
	nx.draw(G, with_labels=True)
	plt.show()


if __name__ == "__main__":
	create_RPS_node()

"""
PYGAME Helper Functions
"""
def AAfilledRoundedRect(surface,color, rect, radius=0.4):
	"""
	Helper function to create a rectangle with rounded corners in PyGame.
	AAfilledRoundedRect(surface,rect,color,radius=0.4)

	surface : destination
	rect    : rectangle
	color   : rgb or rgba
	radius  : 0 <= radius <= 1
	"""

	rect         = Rect(rect)
	color        = Color(*color)
	alpha        = color.a
	color.a      = 0
	pos          = rect.topleft
	rect.topleft = 0,0
	rectangle    = Surface(rect.size,SRCALPHA)

	circle       = Surface([min(rect.size)*3]*2,SRCALPHA)
	draw.ellipse(circle,(0,0,0),circle.get_rect(),0)
	circle       = transform.smoothscale(circle,[int(min(rect.size)*radius)]*2)

	radius              = rectangle.blit(circle,(0,0))
	radius.bottomright  = rect.bottomright
	rectangle.blit(circle,radius)
	radius.topright     = rect.topright
	rectangle.blit(circle,radius)
	radius.bottomleft   = rect.bottomleft
	rectangle.blit(circle,radius)

	rectangle.fill((0,0,0),rect.inflate(-radius.w,0))
	rectangle.fill((0,0,0),rect.inflate(0,-radius.h))

	rectangle.fill(color,special_flags=BLEND_RGBA_MAX)
	rectangle.fill((255,255,255,alpha),special_flags=BLEND_RGBA_MIN)

	return surface.blit(rectangle,pos)