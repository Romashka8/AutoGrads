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

		return [module.accGradParameters() for module in self.modules]

	def __repr__(self):

        string = "".join([str(x) + '\n' for x in self.modules])
        
        return string

    def __getitem__(self, x):
        
        return self.modules.__getitem__(x)

    def train(self):

        """
        Распространяет training параметр по всем модулям
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
