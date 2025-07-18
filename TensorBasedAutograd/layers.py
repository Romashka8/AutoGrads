# ----------------------------------------------------------------------------------------------------------- 

from interface import *
from tensor import Tensor

import numpy as np

# -----------------------------------------------------------------------------------------------------------

class Linear(Layer):

	"""
	Реализация линейного(полносвязного) слоя.
	"""

	def __init__(self, n_inputs, n_outputs):

		super(Linear, self).__init__()
		W = np.random.randn(n_inputs, n_outputs) * np.sqrt(2.0 / (n_inputs))
		self.weight = Tensor(W, autograd = True)
		self.bias = Tensor(np.zeros(n_outputs), autograd = True)

		self.parameters.append(self.weight)
		self.parameters.append(self.bias)

	def forward(self, input):

		return input.mm(self.weight) + self.bias.expand(0, len(input.data))

# -----------------------------------------------------------------------------------------------------------

class Sequential(Layer):

	"""
	Реализует класс-контейнер для объединения нескольких слоев.
	"""

	def __init__(self, layers = list()):

		super(Sequential, self).__init__()
		self.layers = layers

	def add(self, layer):

		self.layers.append(layer)

	def forward(self, input):

		for layer in self.layers:
			input = layer.forward(input)
		return input

	def get_parameters(self):

		params = list()
		for layer in self.layers:
			params += layer.get_parameters()
		return params

# -----------------------------------------------------------------------------------------------------------

class Tanh(Layer):

	"""
	Нелинейная активация.
	"""

	def __init__(self):

		super(Tanh, self).__init__()

	def forward(self, input):

		return input.tanh()

# -----------------------------------------------------------------------------------------------------------

class Sigmoid(Layer):

	"""
	Нелинейная активация.
	"""

	def __init__(self):

		super(Sigmoid, self).__init__()

	def forward(self, input):

		return input.sigmoid()

# -----------------------------------------------------------------------------------------------------------
