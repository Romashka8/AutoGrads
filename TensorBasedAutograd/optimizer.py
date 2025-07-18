# ----------------------------------------------------------------------------------------------------------- 

from interface import *

import numpy as np

# ----------------------------------------------------------------------------------------------------------- 

class SGD(Optimizer):

	"""
	Реализует стохастическую модификацию градиентного спуска.
	"""

	def __init__(self, parameters, alpha = 0.1):

		self.parameters = parameters
		self.alpha = alpha

	def zero(self):

		for p in parameters:
			p.grad.data *= 0

	def step(self, zero = True):

		for p in self.parameters:

			p.data -= p.grad.data * self.alpha

			if zero:
				p.grad.data *= 0

# ----------------------------------------------------------------------------------------------------------- 
