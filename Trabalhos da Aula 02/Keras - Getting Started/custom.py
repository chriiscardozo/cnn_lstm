from keras import backend as K
from keras.engine.topology import Layer
import numpy as np
from keras import initializers

# Implementação da Prelu com alpha como uma variável do aprendizado
class PReLULayer(Layer):

    def __init__(self, name, **kwargs):
        self.name = name
        self.init = initializers.get('zero')
        super(PReLULayer, self).__init__(**kwargs)

    def build(self, input_shape):
        param_shape = list(input_shape[1:])

        # Create a trainable array variable for this layer.
        dtype = K.floatx()
        self.alphas = K.variable(self.init(param_shape),dtype=dtype,name='{}_alphas'.format(self.name))
        self.trainable_weights = [self.alphas]

        super(PReLULayer, self).build(input_shape)

    def call(self, x):
        pos = K.relu(x)
        neg = self.alphas * (x - abs(x)) * 0.5
        return pos + neg

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1])