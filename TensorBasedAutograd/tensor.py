# -----------------------------------------------------------------------------------------------------------

import numpy as np

from interface import *

# -----------------------------------------------------------------------------------------------------------

class Tensor(Tensor):

	def __init__(self, data, autograd=False, creators=None, creation_op=None, id=None):

		"""
		data: данные тензора, массив
		creators: узлы, породившие текущий узел в графе вычислений
		creation_op: при помощи какой операции был порожден текущий узел
		grad: сохранение градиента тензора.
		"""

		self.data = np.array(data)
		self.autograd = autograd
		self.grad = None

		if id is None:
			self.id = np.random.randint(0, 100000)
		else:
			self.id = id

		self.creators = creators
		self.creation_op = creation_op
		self.children = {}

		if creators is not None:
			for c in creators:
				if self.id not in c.children:
					c.children[self.id] = 1
				else:
					c.children[self.id] += 1

	def _all_children_grads_accounted_for(self):

		for id, cnt in self.children.items():
			if cnt != 0:
				return False
		return True

	def backward(self, grad=None, grad_origin=None):

		"""
		Получает на вход градиент и распространяет ошибку по графу сети в обратном направлении.
		"""

		if self.autograd:

			if grad is None:
				grad = Tensor(np.ones_like(self.data))

			if grad_origin is not None:
				if self.children[grad_origin.id] == 0:
					raise Exception("Cannot backprop more than once!")
				else:
					self.children[grad_origin.id] -= 1

			if self.grad is None:
				self.grad = grad
			else:
				self.grad += grad

			"""
			Градиенты не должны иметь своих градиентов
			(в контексте торча - градиент - тензор, у которого автоматически градиент не считается)
			"""
			
			assert grad.autograd == False
			
			"""
			Продолжаем обратное распространение только есть 
			есть что распространять обратно и все градиенты (от детей)
			учтены для переопределения ожидания дочерних элементов, если
			"backprop" был вызван непосредственно для этой переменной
			"""

			if self.creators is not None and (self._all_children_grads_accounted_for() or grad_origin is not None):


				if self.creation_op == "add":

					"""
					В случае с сложением градиент просто распространяется по родительским узлам.
					"""

					self.creators[0].backward(self.grad, self)
					self.creators[1].backward(self.grad, self)

				if self.creation_op == "neg":

					"""
					Для отрицания меняем знак у первого родителя.
					"""

					self.creators[0].backward(self.grad.__neg__())

	def __add__(self, other):

		if self.autograd and other.autograd:

			return Tensor(
				self.data + other.data,
				autograd = True,
				creators = [self, other],
				creation_op = "add"
			)

		return Tensor(self.data + other.data)

	def __neg__(self):

		if self.autograd:

			return Tensor(
				self.data * -1,
				autograd = True,
				creators = [self],
				creation_op = "neg"
			)

		return Tensor(self.data * -1)

	def __repr__(self):

		return str(self.data.__repr__())

	def __str__(self):

		return str(self.data.__str__())

# -----------------------------------------------------------------------------------------------------------

if __name__ == "__main__":

	a = Tensor([1, 2, 3, 4, 5], autograd=True)
	b = Tensor([2, 2, 2, 2, 2], autograd=True)
	c = Tensor([5, 4, 3, 2, 1], autograd=True)

	d = a + (-b)
	e = (-b) + c
	f = d + e

	f.backward(Tensor([1, 1, 1, 1, 1]))
	
	print(b.grad.data == np.array([-2, -2, -2, -2, -2]))	

# -----------------------------------------------------------------------------------------------------------
