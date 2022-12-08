import base

class History(base.History):
	def __init__(self) -> None:
		super().__init__()
	
	def is_terminal(self):
		return super().is_terminal()