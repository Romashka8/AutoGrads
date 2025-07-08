# -----------------------------------------------------------------------------------------------------------

import numpy as np

from interface import *

# -----------------------------------------------------------------------------------------------------------

class Tensor(Tensor):

	def __init__(self, data, creators=None, creation_op=None):

		"""
		data: данные тензора, массив
		creators: узлы, породившие текущий узел в графе вычислений
		creation_op: при помощи какой операции был порожден текущий узел
		grad: сохранение градиента тензора.
		"""

		self.data = np.array(data)
		self.creators = creators
		self.creation_op = creation_op
		self.grad = None

	def backward(self, grad):

		"""
		Получает на вход градиент и распространяет ошибку по графу сети в обратном направлении.
		"""

		self.grad = grad

		if self.creation_op == "add":

			"""
			В случае с сложением градиент просто распространяется по родительским узлам.
			"""

			self.creators[0].backward(grad)
			self.creators[1].backward(grad)

	def __add__(self, other):

		return Tensor(self.data + other.data, creators = [self, other], creation_op = "add")

	def __repr__(self):

		return str(self.data.__repr__())

	def __str__(self):

		return str(self.data.__str__())

# -----------------------------------------------------------------------------------------------------------

if __name__ == "__main__":

	x = Tensor([1, 2, 3, 4, 5])
	y = Tensor([2, 2, 2, 2, 2])

	z = x + y

	z.backward(Tensor([1, 1, 1, 1, 1]))

	print(x.grad)
	print(y.grad)
	print(z.creators)
	print(z.creation_op)

	

# -----------------------------------------------------------------------------------------------------------
