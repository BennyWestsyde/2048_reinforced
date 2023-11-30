
class Cell:
	def __init__(self, value: int):
		self.value = value
		self.setColor()

	def setColor(self):
		# Colors will move through the rainbow as the value increases
		color_dict = {
		2: "\33[45m", # Purple
		4: "\33[0;105m", # Light Purple
		8: "\33[44m", # Blue
		16: "\33[0;104m", # Light Blue
		32: "\33[42m", # Green
		64: "\33[0;102m", # Light Green
		128: "\33[43m", # Yellow
		256: "\33[0;103m", # Light Yellow
		512: "\33[41m", # Red
		1024: "\33[0;101m", # Light Red
		2048: "\33[0;100m" # Gray
		}
		if self.value in color_dict:
			self.color = color_dict[self.value]
			return color_dict[self.value]
		else:
			self.color = "\33[0;100m"
			return "\33[0;100m" # Gray




	def __hash__(self):
		return hash(self.value)

	def __str__(self):
		return str(self.value)

	def __repr__(self):
		return str(self.value)

	def __eq__(self, other):
		if type(other) == Cell:
			return self.value == other.value
		else:
			return self.value == other

	def __lt__(self, other):
		if type(other) == Cell:
			return self.value < other.value
		else:
			return self.value < other

	def __gt__(self, other):
		if type(other) == Cell:
			return self.value > other.value
		else:
			return self.value > other

	def __le__(self, other):
		if type(other) == Cell:
			return self.value <= other.value
		else:
			return self.value <= other

	def __ge__(self, other):
		if type(other) == Cell:
			return self.value >= other.value
		else:
			return self.value >= other

	def __ne__(self, other):
		if type(other) == Cell:
			return self.value != other.value
		else:
			return self.value != other
		
	def __add__(self, other):
		if type(other) == Cell or type(other) == EmptyCell:
			self.value += other.value
			self.setColor()
		else:
			self.value += other
			self.setColor()
		return self
		
			
	def __int__(self):
		return self.value

	

class EmptyCell(Cell):
	def __init__(self):
		super().__init__(0)

