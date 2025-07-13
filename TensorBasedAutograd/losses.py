# ----------------------------------------------------------------------------------------------------------- 

import numpy as np

from interface import *
from tensor import Tensor

# -----------------------------------------------------------------------------------------------------------

class MSELoss(Layer):

	def __init__(self):

		super(MSELoss, self).__init__()

	def forward(self, pred, target):

		return ((pred - target) * (pred - target)).sum(0)

# ----------------------------------------------------------------------------------------------------------- 
