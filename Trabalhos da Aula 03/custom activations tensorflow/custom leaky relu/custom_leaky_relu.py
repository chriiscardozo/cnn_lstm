'''
Implementação da função leaky relu e todas suas primitivas usando apenas python
Links de discussão: 
	https://stackoverflow.com/questions/39921607/how-to-make-a-custom-activation-function-with-only-python-in-tensorflow
	https://github.com/tensorflow/tensorflow/issues/1095
	https://gist.github.com/harpone/3453185b41d8d985356cbe5e57d67342
'''

import numpy as np
import tensorflow as tf

def leaky_relu(x): # primeira versão com parâmetro a constante
	a = 0.01 # Leaky Relu
	if(x < 0): return a * x
	else: return x

def d_leaky_relu(x):
	a = 0.01
	if(x < 0): return a
	else: return 1

np_leaky_relu = np.vectorize(leaky_relu)
np_d_leaky_relu = np.vectorize(d_leaky_relu)

# funções no TF usam float32 e np usam float64
np_d_leaky_relu32 = lambda x: np_d_leaky_relu(x).astype(np.float32)
np_leaky_relu32 = lambda x: np_leaky_relu(x).astype(np.float32)

'''
py_func documentação: https://www.tensorflow.org/api_docs/python/tf/py_func
Params:
	func: função em python que recebe uma lista de NumPy ndarray
	inp: lista de objetos do tipo Tensor
	Tout: lista de tipo de retornos da função
	stateful: funções stateless retornam sempre o mesmo valor para a mesma entrada
	name: nome da operação

O TF não sabe como calcular o gradiente ao registrar uma nova função.
Por isso o hack abaixo é feito no momento da criação da função para que o
TF saiba como calcular o gradiente também.
Hack de: https://gist.github.com/harpone/3453185b41d8d985356cbe5e57d67342
Doc. de gradientes no TF:
	https://www.tensorflow.org/versions/r0.11/api_docs/python/framework.html#RegisterGradient
	https://www.tensorflow.org/versions/r0.11/api_docs/python/framework.html
'''

def py_func(func, inp, Tout, stateful=True, name=None, grad=None):

    # Need to generate a unique name to avoid duplicates:
    rnd_name = 'PyFuncGrad' + str(np.random.randint(0, 1E+8))

    tf.RegisterGradient(rnd_name)(grad)  # see _MySquareGrad for grad example
    g = tf.get_default_graph()
    with g.gradient_override_map({"PyFunc": rnd_name}):
        return tf.py_func(func, inp, Tout, stateful=stateful, name=name)

def tf_d_leaky_relu(x, name=None):
	with tf.name_scope(name, "d_leaky_relu", [x]) as name:
		y = tf.py_func(np_d_leaky_relu32,
					[x],
					[tf.float32],
					name=name,
					stateful=False)
		return y[0]

# função que calcula o gradiente
def leaky_relu_grad(op, grad):
	x = op.inputs[0]

	n_gr = tf_d_leaky_relu(x)
	return grad * n_gr
	'''
	se a função de ativação tiver mais de um parâmetro,
	precisamos retornar uma tupla para cada input.
	Por exemplo, se a função fosse a - b, precisaríamos retornar
	as derivadas em relação a 'a' (+1) e 'b'(-1), que seria:
	return +1*grad, -1*grad
	'''

def tf_leaky_relu(x, name=None):
	with tf.name_scope(name, "leaky_relu", [x]) as name:
		y = py_func(np_leaky_relu32,
					[x],
					[tf.float32],
					name=name,
					grad=leaky_relu_grad)
		return y[0]


with tf.Session() as sess:
	x = tf.constant([-0.25, -0.1, 0.0, 0.2, 0.7, 1.2, 2.0])
	y = tf_leaky_relu(x)
	tf.global_variables_initializer().run()
	print(x.eval())
	print(y.eval())
	print(tf.gradients(y, [x])[0].eval())