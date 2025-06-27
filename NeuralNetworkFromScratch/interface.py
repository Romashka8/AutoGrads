import abc
import numpy as np


# ----------------------------------------------------------------------------------------------------------------------------

class Module(abc.ABC):

	"""
	Можно думать о модуле как о некотором черном ящике,
	который может преобразовывать 'input' данные в 'output' данные.
	Это аналогично применению к данных функции, которая называется 'forward':

		output = module.forward(input)

	Модуль также должен быть способен осуществлять backward pass: искать производную 'forward' функции.
	Более того, модуль должен уметь находить ее производную даже тогда, когда функция является правилом цепочки(chain rule).
	Последнее подразумевает, что существует градиент с предыдущего шага chain rule(то есть слой до этого - дифференцируем).

		gradInput = module.backward(input, gradOutput) # gradOutput - с предыдущего слоя
	"""

	def __init__(self):

		self.output = None
		self.gradInput = None
		self.training = True

	def forward(self, input):

		"""
		Принимает входной объект и вычисляет соответствующий выход для модуля.
		"""

		return self.updateOutput(input)

	def backward(self, input, gradOutput):
		
		"""
		Выполняет backpropagation шаг для всего модуля относительно заданного входного сигнала

		Включает в себя:
		 - вычисление градиента с использованием 'input' данных (это нужно для дальнейшего обратного распространения),
		 - вычисление градиента с параметрами (для обновления параметров при оптимизации)
        """

		self.updateGradInput(input, gradOutput)
		self.accGradParameters(input, gradOutput)

		return self.gradInput

	@abc.abstractmethod
	def updateOutput(self, input):

		"""
		Вычисляет 'output' используя текущий набор параметров класса и input данные.
		Результат возвращаемый этой функцией сохраняется в 'output' поле.

		Нужно убедиться, что данные как сохраняются в output, так и возвращаются.
        """

        # Самый простой случай:

        # self.output = input
        # return self.output

		pass

	@abc.abstractmethod
	def updateGradInput(self, input, gradOutput):

		"""
		Вычисляет градиент модуля с учетом его собственного 'input'.
		Сохраняетсяв 'gradInput'. Также, состояние 'gradInput' обновляется соответственно.

		Размерность 'gradInput' всегда совпадает с размерностью 'input'.

		Нужно убедиться, что градиенты сохраняются в поле 'gradInput' и возвращаются из него же.
		"""

		# Самый простой случай

		# self.gradInput = gradOutput
		# return self.gradInput

		pass

	def accGradParameters(self, input, gradOutput):

		"""
		Вычисляет градиент модуля с учетом его собственных параметров.
		Нет необходимости переопределять, если модуль не имеет параметров (по типу ReLU)
		"""

		pass

	def zeroGradParameters(self):
		
		"""
		Зануляет 'gradParams' переменную, если модуль имеет соответствующие параметры
		"""
		
		pass

	def getParameters(self):

		"""
		Возвращает список с параметрами модуля.
		Если у модуля нет параметров - возвращает пустой список.
		"""

		return []

	def getGradParameters(self):

		"""
		Возвращает список с градиентами относительно его параметров.
		Если модуль не имеет параметров, возвращается пустой список.
		"""

		return []

	def train(self):

		"""
		Переводит модель в тренировочный режим.
        Поведение на train/test отличается для Dropout, BatchNorm.
		"""

		self.training = True

	def evaluate(self):
		"""
		Переводит модель в eval(оценочный) режим.
		Поведение на train/test отличается для Dropout, BatchNorm.
		"""

		self.training = False

# ----------------------------------------------------------------------------------------------------------------------------

class Criterion(abc.ABC):

	"""
	Критерии используются для скора ответа моделей.
	"""

	def __init__(self):
		
		self.output = None
		self.gradInput = None

	def forward(self, input, target):

		"""
		Принимает input и target, считает функцию потерь, которая была реализована в критерии и возвращает результат.
		Эту функцию не стоит переопределять. Вместо этого переопределяется updateOutput
		"""

		return return self.updateOutput(input, target)

	def backward(self, input, target):

		"""
		Принимает input и target, считает градиент функции потерь, которая была реализована в критерии и возвращает результат.
		Эту функцию не стоит переопределять. Вместо этого переопределяется updateGradInput
		"""

		return self.updateGradInput(input, target)

	@abc.abstractmethod
	def updateOutput(self, input, target):
        pass

    @abc.abstractmethod
    def updateGradInput(self, input, target):
        pass

    def __repr__(self):
        
        """
        Следует переопределять в каждм модуле, если хотим получать красивую печать.
        """
        
        return 'Criterion'
