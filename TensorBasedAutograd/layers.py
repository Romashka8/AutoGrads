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
