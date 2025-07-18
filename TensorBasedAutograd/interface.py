# -----------------------------------------------------------------------------------------------------------

import numpy as np

import abc

# ----------------------------------------------------------------------------------------------------------- 

class Tensor(abc.ABC):

	@abc.abstractmethod
	def __init__(self):

		"""
		Инициализация тензора вместе с данными.
		"""

		pass

	@abc.abstractmethod
	def backward(self):

		"""
		Реализация обратного распространения ошибки.
		"""

		pass

	@abc.abstractmethod
	def __add__(self):

		"""
		Реализация сложения тензоров.
		"""

		pass

	@abc.abstractmethod
	def __neg__(self):

		"""
		Реализация отрицания элементов тензора.
		"""

		pass

	@abc.abstractmethod
	def __sub__(self):

		"""
		Реализация тензорного вычитания.
		"""

		pass

	@abc.abstractmethod
	def __mul__(self):

		"""
		Реализует умножение тензоров.
		"""

		pass


	@abc.abstractmethod
	def sum(self):

		"""
		Суммирует тензоры по определенному измерению.
		"""

		pass

	@abc.abstractmethod
	def expand(self):

		"""
		Расширяет тензор.
		"""

		pass

	@abc.abstractmethod
	def transpose(self):

		"""
		Транспонирует тензоры.
		"""

		pass

	@abc.abstractmethod
	def mm(self):

		"""
		Матричное умножение тензоров.
		"""

		pass

	@abc.abstractmethod
	def __repr__(self):
		
		"""
		Вывод информации о тензоре.
		"""

		pass

	@abc.abstractmethod
	def __str__(self):
		
		"""
		Вывод информации о тензоре.
		"""

		pass

# ----------------------------------------------------------------------------------------------------------- 

class Optimizer(abc.ABC):

	@abc.abstractmethod
	def __init__(self):

		"""
		Инициализация оптимизатора.
		"""

		pass

	@abc.abstractmethod
	def step(self):

		"""
		Реализация шага оптимизатора.
		"""

		pass

# ----------------------------------------------------------------------------------------------------------- 

class Layer(abc.ABC):

	def __init__(self):

		"""
		Инициализация списка параметров.
		"""

		self.parameters = list()

	def get_parameters(self):

		"""
		Получение параметров.
		"""

		return self.parameters

	@abc.abstractmethod
	def forward(self):

		"""
		Реализация прямого распространения ошибки для каждого слоя.
		"""

		pass	

# ----------------------------------------------------------------------------------------------------------- 
