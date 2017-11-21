from keras import backend as K
from keras.engine.topology import Layer
import numpy as np
from keras import initializers
from keras.engine import InputSpec

class BHSA(Layer):

	def __init__(self,
		lambda_initializer=initializers.Constant(value=1),
		t_initializer=initializers.glorot_normal(),
		shared_axes=None, **kwargs):
		super(BHSA, self).__init__(**kwargs)

		self.lambda_initializer = initializers.get(lambda_initializer)
		self.t_initializer = initializers.get(t_initializer)

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

		# lambda
		self.l = self.add_weight(name='l', shape=param_shape, initializer=self.lambda_initializer)
		# tau 1
		self.t = self.add_weight(name='t', shape=param_shape, initializer=self.t_initializer)

		axes = {}
		if(self.shared_axes):
			for i in range(1, len(input_shape)):
				if(i not in self.shared_axes):
					axes[i] = input_shape[i]

		self.input_spec = InputSpec(ndim=len(input_shape), axes=axes)
		self.built = True

	def call(self, inputs):
		# bi-hiperbolic simetric adaptative
		# bhsa(l, t) = h1 - h2
		# h1 = sqrt(( (l^2)*((x + (1/(2*l)))^2) ) + t^2)
		# h2 = sqrt(( (l^2)*((x + (1/(2*l)))^2) ) + t^2)
		h1 = K.sqrt( (K.square(self.l)*K.square(inputs + (1.0/(4*self.l)))) + K.square(self.t) )
		h2 = K.sqrt( (K.square(self.l)*K.square(inputs - (1.0/(4*self.l)))) + K.square(self.t) )
		return (h1 - h2) + 0.5
