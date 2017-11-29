# Credits to: https://github.com/jiamings/ais

from keras import backend as K
import tensorflow as K
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

def K_log_mean_exp(x, axis=None):
	m = K.reduce_max(x, axis=axis, keep_dims=True)
	return m + K.log(K.reduce_mean(K.exp(x - m), axis=axis, keep_dims=True))


def K_parzen(x, mu, sigma):
	d = (K.expand_dims(x, 1) - K.expand_dims(mu, 0)) / sigma
	e = K_log_mean_exp(-0.5 * K.reduce_sum(K.multiply(d, d), axis=2), axis=1)
	e = K.squeeze(e, axis=1)
	z = K.to_float(K.shape(mu)[1]) * K.log(sigma * np.sqrt(np.pi * 2.0))
	return e - z

class ParsenDensityEstimator(object):
	def __init__(self):
		self.x = K.placeholder(K.float32)
		self.mu = K.placeholder(K.float32)
		self.sigma = K.placeholder(K.float32, [])
		self.ll = K_parzen(self.x, self.mu, self.sigma)

	def logpdf(self, x, mu, sigma, sess):
		sess = sess or self.sess
		return sess.run(self.ll, feed_dict={self.x: x, self.mu: mu, self.sigma: sigma})

	def get_ll(self, x, mu, sigma, sess, batch_size=1000):
		lls = []
		inds = range(x.shape[0])
		n_batches = int(np.ceil(float(len(inds)) / batch_size))
		for i in range(n_batches):
			lls.extend(self.logpdf(x[inds[i::n_batches]], mu, sigma, sess))
		return np.array(lls)
