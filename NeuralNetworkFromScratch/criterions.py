import numpy as np

from interface import *


# ----------------------------------------------------------------------------------------------------------------------------

class ClassNLLCriterion(Criterion):

	"""
	Реализация Negative LogLikelihood criterion.
	"""

    def __init__(self, epsilon=1e-15):

    	super(ClassNLLCriterion, self).__init__()

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

        return 'Class Negative Log Likelihood Criterion'

# ----------------------------------------------------------------------------------------------------------------------------
