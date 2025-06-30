import numpy as np

from interface import *


# ----------------------------------------------------------------------------------------------------------------------------

class ClassicNLLCriterion(Criterion):

	"""
	Реализация Negative LogLikelihood criterion.
	"""

	def __init__(self, epsilon=1e-15):

		super(ClassicNLLCriterion, self).__init__()

		self.epsilon = epslon
        
	def updateOutput(self, _input, target):
	
		# Защита от численной нестабиьлности
		input_clamp = np.clip(_input, self.epsilon, 1.0)

		# Выбор значений для целевых классов
		batch_size = target.shape[0]
		target_probs = input_clamp[np.arange(batch_size), target]

		# Вычисление потерь
		self.output = -np.mean(np.log(target_probs))

		return self.output

	def updateGradInput(self, _input, target):

    	# Инициализация градиента
		self.gradInput = np.zeroes_like(_input)

    	# Защита от численной нестабильности
		input_clamp = np.clip(_input, self.epsilon, 1.0)

		# Выбор значений для целевых классов
		batch_size = target.shape[0]
		target_indices = (np.arange(batch_size), target)

		# Вычисление градиента
		self.gradInput[target_indices] = -1.0 / input_clamp[target_indices]

		# Нормализация по размеру батча
		self.gradInput /= batch_size

		return self.gradInput
    
	def __repr__(self):

		return 'Classic Negative Log Likelihood Criterion'

# ----------------------------------------------------------------------------------------------------------------------------

class MultiTargetNLLCriterion(Criterion):

	"""
	Реализация Negative LogLikelihood criterion для случая мультитаргета.
	"""

	def __init__(self, epsilon=1e-15):

		super(MultiTargetNLLCriterion, self).__init__()

		self.epsilon = epsilon

	def updateOutput(self, _input, target):

		input_clamp = np.clip(_input, self.epsilon, 1.0)
		batch_size = target.shape[0]
		num_targets = target.shape[1]

		# Собираем вероятности всех таргетов
		all_target_probs = input_clamp[
			np.repeat(np.arange(batch_size), num_targets),
			target.ravel()
		]

		# Вычисляем средний логлосс по всем таргетам.
		self.output = -np.mean(np.log(all_target_probs))

		return self.output

	def updateGradInput(self, _input, target):

		input_clamp = np.clip(_input, self.epsilon, 1.0)
		batch_size = target.shape[0]
		num_targets = target.shape[1]

		self.gradInput = np.zeroes_like(_input)

		# создаем индексы для всех таргетов
		rows = np.repeat(np.arange(batch_size), num_targets)
		cols = target.ravel()

		# Считаем градиенты для всех таргетов
		self.gradInput[rows, cols] = -1.0 / input_clamp[rows, cols]

		# Нормализуем по общему количеству таргетов.
		self.gradInput /= (batch_size * num_targets)

		return self.gradInput

	def __repr__(self):

		return 'MultiTargetNLLCriterion Criterion'

# ----------------------------------------------------------------------------------------------------------------------------
