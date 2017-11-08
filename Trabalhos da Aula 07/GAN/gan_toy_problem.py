# Referencia
#	http://blog.aylien.com/introduction-generative-adversarial-networks-code-tensorflow/

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.stats.kde import gaussian_kde

def gaussianSamples(mean, std, N):
	data = np.random.normal(mean, std, N)
	data.sort()
	return np.float32(data)

def noiseSamples(range_data, N):
	data = np.linspace(-range_data, range_data, N) + np.random.random(N) * 0.01
	return np.float32(data)

def plotData(data):
	kde = gaussian_kde( data )
	dist_space = np.linspace( min(data), max(data), 100 )
	plt.plot( dist_space, kde(dist_space) )

def linear(input, output_dim, scope=None, stddev=1.0):
	with tf.variable_scope(scope or 'linear'):
		w = tf.get_variable(
			'w',
			[input.get_shape()[1], output_dim],
			initializer=tf.random_normal_initializer(stddev=stddev)
		)
		b = tf.get_variable(
			'b',
			[output_dim],
			initializer=tf.constant_initializer(0.0)
		)
		return tf.matmul(input, w) + b

def generatorNetwork(input, hidden_size):
	h0 = tf.nn.softplus(linear(input, hidden_size, 'g0'))
	h1 = linear(h0, 1, 'g1')
	return h1

def discriminatorNetwork(input, h_dim, minibatch_layer=True):
	h0 = tf.nn.relu(linear(input, h_dim * 2, 'd0'))
	h1 = tf.nn.relu(linear(h0, h_dim * 2, 'd1'))
	h2 = tf.nn.relu(linear(h1, h_dim * 2, scope='d2'))
	h3 = tf.sigmoid(linear(h2, 1, scope='d3'))

	return h3

def optimizer(loss, var_list):
	learning_rate = 0.001
	step = tf.Variable(0, trainable=False)
	optimizer = tf.train.AdamOptimizer(learning_rate).minimize(
		loss,
		global_step=step,
		var_list=var_list
	)
	return optimizer

def log(x):
	return tf.log(tf.maximum(x, 1e-5))

class GAN(object):
	def __init__(self, hidden_size, batch_size):
		with tf.variable_scope('G'):
			self.z = tf.placeholder(tf.float32, shape=(batch_size, 1))
			self.G = generatorNetwork(self.z, hidden_size)

		self.x = tf.placeholder(tf.float32, shape=(batch_size, 1))
		with tf.variable_scope('D'):
			self.D1 = discriminatorNetwork(
				self.x,
				hidden_size
			)
		with tf.variable_scope('D', reuse=True):
			self.D2 = discriminatorNetwork(
				self.G,
				hidden_size
			)

		self.loss_d = tf.reduce_mean(-log(self.D1) - log(1 - self.D2))
		self.loss_g = tf.reduce_mean(-log(self.D2))

		vars = tf.trainable_variables()
		self.d_params = [v for v in vars if v.name.startswith('D/')]
		self.g_params = [v for v in vars if v.name.startswith('G/')]

		self.opt_d = optimizer(self.loss_d, self.d_params)
		self.opt_g = optimizer(self.loss_g, self.g_params)

def samples(
	model,
	session,
	mean,
	std,
	sample_range,
	batch_size,
	num_points=10000,
	num_bins=100
):
	'''
	Return a tuple (db, pd, pg), where db is the current decision
	boundary, pd is a histogram of samples from the data distribution,
	and pg is a histogram of generated samples.
	'''
	xs = np.linspace(-sample_range, sample_range, num_points)
	bins = np.linspace(-sample_range, sample_range, num_bins)

	# decision boundary
	db = np.zeros((num_points, 1))
	for i in range(num_points // batch_size):
		db[batch_size * i:batch_size * (i + 1)] = session.run(
			model.D1,
			{
				model.x: np.reshape(
					xs[batch_size * i:batch_size * (i + 1)],
					(batch_size, 1)
				)
			}
		)

	# data distribution
	d = gaussianSamples(mean, std, num_points)
	pd, _ = np.histogram(d, bins=bins, density=True)

	# generated samples
	zs = np.linspace(-sample_range, sample_range, num_points)
	g = np.zeros((num_points, 1))
	for i in range(num_points // batch_size):
		g[batch_size * i:batch_size * (i + 1)] = session.run(
			model.G,
			{
				model.z: np.reshape(
					zs[batch_size * i:batch_size * (i + 1)],
					(batch_size, 1)
				)
			}
		)
	pg, _ = np.histogram(g, bins=bins, density=True)

	return db, pd, pg


def plot_distributions(samps, sample_range):
	db, pd, pg = samps
	#db_x = np.linspace(-sample_range, sample_range, len(db))
	p_x = np.linspace(-sample_range, sample_range, len(pd))
	f, ax = plt.subplots(1)
	#ax.plot(db_x, db, label='decision boundary')
	ax.set_ylim(0, 1)
	plt.plot(p_x, pd, label='real data')
	plt.plot(p_x, pg, label='generated data')
	plt.title('1D Generative Adversarial Network')
	plt.xlabel('Data values')
	plt.ylabel('Probability density')
	plt.legend()
	plt.show()

def main():
	default_range = 8
	batch_size = 8
	mean = 4
	std = 0.5
	hidden_size = 4

	model = GAN(hidden_size, batch_size)

	with tf.Session() as sess:
		tf.local_variables_initializer().run()
		tf.global_variables_initializer().run()

		for step in range(5000):
			#update discriminator
			x = gaussianSamples(mean, std, batch_size)
			z = noiseSamples(default_range, batch_size)

			loss_d, _, = sess.run([model.loss_d, model.opt_d], {
				model.x: np.reshape(x, (batch_size, 1)),
				model.z: np.reshape(z, (batch_size, 1))
			})

			# update generator
			z = noiseSamples(8, batch_size)
			loss_g, _ = sess.run([model.loss_g, model.opt_g], {
				model.z: np.reshape(z, (batch_size, 1))
			})

			if step % 500 == 0:
				print('{}: {:.4f}\t{:.4f}'.format(step, loss_d, loss_g))
				samps = samples(model, sess, mean, std, default_range, batch_size)
				plot_distributions(samps, default_range)

if __name__ == '__main__':
	main()