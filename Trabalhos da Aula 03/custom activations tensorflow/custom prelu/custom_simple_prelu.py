import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


def tf_prelu(alpha, x):
	cond = tf.less(x, tf.constant(0.0))
	return tf.where(cond, alpha*x, x)


# ****************************************************************
# Código desenvolvido no trabalho da aula 01 para MNIST com CNN
# Mudança: uso da função prelu com parâmetros no aprendizado
# ****************************************************************
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

import tensorflow as tf

x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])

def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)

def bias_variable(shape):
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial)

def conv2d(x, W):
	# padding = SAME mantem dimensoes da img
	return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
	# padding = SAME mantem dimensoes da img
	return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# segundo e terceira dimensoes sao W e H do input e a quarta é a qtd de canais
x_image = tf.reshape(x, [-1, 28, 28, 1])

alpha1 = tf.Variable([0.], tf.float32)
alpha2 = tf.Variable([-.5], tf.float32)
alpha3 = tf.Variable([1.8], tf.float32)

# *** Primeira layer ***
# Filtro 5 x 5; 1 canal na entrada; 32 canais de saída (filtros)
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

h_conv1 = tf_prelu(alpha1, conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# *** Segunda layer ***
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf_prelu(alpha2, conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)


# *** FC layer ***
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf_prelu(alpha3, tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# *** Dropout layer ***
# Essa camada busca reduzir os casos de overfitting
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# *** output layer ***
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2


# *** Treino e avaliação dos resultados ***
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	for i in range(20000):
		batch = mnist.train.next_batch(50)
		if i % 100 == 0:
			train_accuracy = accuracy.eval(feed_dict={
					x: batch[0], y_: batch[1], keep_prob: 1.0})
			print('step %d, training accuracy %g' % (i, train_accuracy))
			v1 = sess.run(alpha1)
			v2 = sess.run(alpha2)
			v3 = sess.run(alpha3)
			print(v1, v2, v3)
		train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

	print('test accuracy %g' % accuracy.eval(feed_dict={
			x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))

# ****************************************************************
#
# ****************************************************************