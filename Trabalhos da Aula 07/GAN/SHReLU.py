from keras import backend as K
from keras.engine.topology import Layer
import numpy as np
from keras import initializers
from keras.engine import InputSpec

class SHReLU(Layer):

	def __init__(self, alpha_initializer='zeros', shared_axes=None, **kwargs):
		super(SHReLU, self).__init__(**kwargs)

		self.alpha_initializer = initializers.get(alpha_initializer)

		if(shared_axes == None):
			self.shared_axes = None
		elif not isinstance(shared_axes, (list, tuple)):
			self.shared_axes = [shared_axes]
		else:
			self.shared_axes = list(shared_axes)

	def build(self, input_shape):
		param_shape = list(input_shape[1:])
		self.param_broadcast = [False] * len(param_shape)
		if self.shared_axes is not None:
			for i in self.shared_axes:
				param_shape[i - 1] = 1
				self.param_broadcast[i - 1] = True

		self.alpha = self.add_weight(name='alpha', shape=param_shape, initializer=self.alpha_initializer)

		axes = {}
		if(self.shared_axes):
			for i in range(1, len(input_shape)):
				if(i not in self.shared_axes):
					axes[i] = input_shape[i]

		self.input_spec = InputSpec(ndim=len(input_shape), axes=axes)
		self.built = True

	def call(self, inputs):
		# shrelu(x,alpha) = (x + sqrt(x^2 + alpha^2)/2)
		return (inputs + K.sqrt(K.square(inputs) + K.square(self.alpha)))/2.0