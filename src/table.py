# Lookup table mapping an int -> position of "1" bit
# Ex: 4 (100) -> 2

def generate_table():
	TABLE = {}
	# Create the table
	val = 1
	for i in range(64):
		TABLE[val] = i
		val *= 2
	
	return TABLE