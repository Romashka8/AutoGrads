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

class Embedding(Layer):

	"""
	Слой эмбеддинга для NLP задач.
	"""

	def __init__(self, vocab_size, dim):

		super(Embedding, self).__init__()

		self.vocab_size = vocab_size
		self.dim = dim

		# Этот тип рандомной инициализации нужен для согласования с word2vec.
		self.weight = Tensor((np.random.randn(vocab_size, dim) - 0.5) / dim, autograd = True)

		self.parameters.append(self.weight)

	def forward(self, input):

		return self.weight.index_select(input)

# -----------------------------------------------------------------------------------------------------------

class CrossEntropyLoss(Layer):

	"""
	Cross Entropy слой.
	"""

	def __init__(self):

		super(CrossEntropyLoss, self).__init__()

	def forward(self, input, target):

		return input.cross_entropy(target)

# -----------------------------------------------------------------------------------------------------------

class RNNCell(Layer):

	"""
	Реализация RNN слоя.
	"""

	def __init__(self, n_inputs, n_hidden, n_output, activation = "sigmoid"):

		super().__init__(RNNCell, self).__init__()

		self.n_inputs = n_inputs
		self.n_hidden = n_hidden
		self.n_output = n_output

		if activation == "sigmoid":
			self.activation = Sigmoid()
		elif activation == "tanh":
			self.activation = Tanh()
		else:
			raise Exception("Non-linearity not found!")

		self.w_ih = Linear(n_inputs, n_hidden)
		self.w_hh = Linear(n_hidden, n_hidden)
		self.w_ho = Linear(n_hidden, n_output)

		self.parameters += self.w_ih.get_parameters()
		self.parameters += self.w_hh.get_parameters()
		self.parameters += self.w_ho.get_parameters()

	def forward(self, input, hidden):

		from_prev_hidden = self.w_hh.forward(hidden)
		combined = self.w_ih.forward(input) + from_prev_hidden
		new_hidden = self.activation.forward(combined)
		output = self.w_ho.forward(new_hidden)

		return output, new_hidden

	def init_hidden(self, batch_size = 1):

		return Tensor(np.zeros((batch_size, self.n_hidden)), autograd = True)

# -----------------------------------------------------------------------------------------------------------
