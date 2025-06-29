import numpy as np

from interface import *


# ----------------------------------------------------------------------------------------------------------------------------

class Sequential(Module):
	
	"""
	В этом классе реализуется контейнер, который последовательно преобразует 'input' данные.
	'input' обрабатывается каждым модулем(слоем) в self.modules последовательно.
	Итоговый массив называется 'output'
	"""

	def __init__(self):

		super(Sequential, self).__init__()
		self.modules = []

	def add(self, module):
		
		"""
		Добавляет модуль в контейнер.
		"""

		self.modules.append(module)

	def updateOutput(self, input):

		"""
		Стандартный workflow forward pass-а:

			y_0 = module[0].forward(input)
			y_1 = module[1].forward(y_0)
			...
			output = module[n-1].forward(y_{n-2})

		Просто напишем небольшой цикл
		"""

		self.output = input

		for module in self.modules:
			self.output = module.forward(self.output)

		return self.output

	def backward(self, input, gradOutput):

		"""
		Workflow backward pass-а:

            g_{n-1} = module[n-1].backward(y_{n-2}, gradOutput)
            g_{n-2} = module[n-2].backward(y_{n-3}, g_{n-1})
            ...
            g_1 = module[1].backward(y_0, g_2)
            gradInput = module[0].backward(input, g_1)

		В каждый модуль нужно передать input, который модуль видел на forward pass-е.
		Это используется во время вычисления градиентов.
		Нужно быть увереным, что input для 'i-ого' слоя - output  'module[i]' (просто тот же input, что и на forward pass)
		и НЕ 'input' для этого Sequential модуля!
		"""

		for i in range(len(self.modules) - 1, 0, -1):
			gradOutput = self.modules[i].backward(self.modules[i - 1].output, gradOutput)

		self.gradInput = self.modules[0].backward(input, gradOutput)

		return self.gradInput

	def zeroGradParameters(self):

		for module in self.modules:
			module.zeroGradParameters()

	def getParameters(self):

		"""
		Возвращает список всех параметров.
		"""

		return [module.getParameters() for module in self.modules]

	def __repr__(self):
		
		string = "".join([str(x) + '\n' for x in self.modules])

		return 'Sequential:\n' + string

	def __getitem__(self, x):

		return self.modules.__getitem__(x)

	def train(self):

		"""
		Распространяет training по всем модулям. 
		"""

		self.training = True

		for module in self.modules: module.train()

	def evaluate(self):

		"""
		Распространяет training параметр по всем модулям
		"""

		self.training = False

		for module in self.modules: module.evaluate()

# ----------------------------------------------------------------------------------------------------------------------------

class Linear(Module):

	"""
	Модуль, который применяет линейное преобразование.
	Его также называют полносвязным слоем.

	Модуль работает с 2D input-ом размера (n_samples, n_features).
	"""

	def __init__(self, n_in, n_out):
		
		super(Linear, self).__init__()

		# Неплохой вариант инициализации
		stdv = 1. / np.sqrt(n_in)
		self.W = np.random.uniform(-stdv, stdv, size = (n_out, n_in))
		self.b = np.random.uniform(-stdv, stdv, size = n_out)

		self.gradW = np.zeros_like(self.W)
		self.gradb = np.zeros_like(self.b)

	def updateOutput(self, _input):
		
		self.output = np.dot(_input, self.W) + self.b

		return self.output

	def updateGradInput(self, _input, gradOutput):

		self.gradInput = np.dot(gradOutput, self.W.T)

		return self.gradInput

	def accGradParameters(self, _input, gradOutput):

		"""
		Необходимо для сложных оптимизаторов. 
		delta_W = input.T @ delta_y
		delta_y - она же и gradOutput
		"""

		self.gradW += np.dot(_input.T, gradOutput)
		self.gradb += np.sum(gradOutput, axis=0)

	def zeroGradParameters(self):

		self.gradW.fill(0)
		self.gradb.fill(0)

	def getParameters(self):

		return [self.W, self.b]

	def getGradParameters(self):

		return [self.gradW, self.gradb]

	def __repr__(self):

		s = self.W.shape
		q = 'Linear %d -> %d' %(s[1], s[0])

		return q

# ----------------------------------------------------------------------------------------------------------------------------

class SoftMax(Module):

	"""
	SoftMax модуль - используется в качестве выходного модуля
	в задачах многоклассовой классификации.
	"""

	def __init__(self):

		super(SoftMax, self).__init__()

	def updateOutput(self, _input):

		# Начнем с нормализации для численной стабильности
		self.output = np.subtract(_input, _input.max(axis=1, keepdims=True))

		self.output = np.exp(self.output)
		self.output = self.output / np.sum(self.output, axis=1, keepdims=True)

		return self.output

	def updateGradInput(self, _input, gradOutput):

		# Расчитаем градиент:
		dot_product = np.sum(self.output * gradOutput, axis=1, keepdims=True)
		self.gradInput = self.output * (gradOutput - dot_product)

		return self.gradInput

	def __repr__(self):

		return 'SoftMax layer'

# ----------------------------------------------------------------------------------------------------------------------------

class LogSoftMax(Module):

	"""
	Применяет Log(SoftMax(x)) к входным данным.
	"""

	def __init__(self):

		super(LogSoftMax, self).__init__()

	def updateOutput(self, _input):

		# Для начала стабилизируем данные
		shifted = _input - _input.max(axis=1, keepdims=True)

		# Вычисление log(sum(exp)) с защитой от переполнения
		log_sum_exp = np.log(np.sum(np.exp(shifted), axis=1, keepdims=True))
		
		# LogSoftmax = shifted - log(sum(exp))
		self.output = shifted - log_sum_exp

		return self.output

	def updateGradInput(self, _input, gradOutput):

		# Вычисляем softmax из log softmax
		softmax = np.exp(self.output)
		
		# Сумма градиентов по каждому примеру в батче
		sum_grad = np.sum(gradOutput, axis=1, keepdims=True)
		
		# Вычисляем градиент
		self.gradInput = gradOutput - softmax * sum_grad

		return self.gradInput

	def __repr__(self):

		return 'LogSoftMax Layer'

# ----------------------------------------------------------------------------------------------------------------------------

class ReLU(Module):

	def __init__(self):

		super(ReLU, self).__init__()

	def updateOutput(self, _input):

		self.output = np.maximum(_input, 0)

		return self.output

	def updateGradInput(self, _input, gradOutput):

		self.gradInput = np.multiply(gradOutput, _input > 0)

		return self.gradInput

	def __repr__(self):

		return 'ReLU Layer'

# ----------------------------------------------------------------------------------------------------------------------------
