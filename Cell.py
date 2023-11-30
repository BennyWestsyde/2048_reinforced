
class Cell:
	def __init__(self, value: int):
		self.value = value

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
		else:
			self.value += other
		return self
		
			
	def __int__(self):
		return self.value

	

class EmptyCell(Cell):
	def __init__(self):
		super().__init__(0)

