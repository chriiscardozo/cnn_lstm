from GAN import *
import Util
import numpy as np
import time
import sys
import tensorflow as tf
from keras import backend as K

def main():
	start = time.time()

	np.random.seed(42)
	X_train, y_train, X_test, y_test = Util.load_mnist_data()

	output_dir = 'output_default'
	if len(sys.argv) == 2:
		output_dir = sys.argv[1]
	Util.reset_dir(output_dir)

	sess = tf.Session()
	K.set_session(sess)

	gan = GAN(session=sess)
	gan.train(X_train, X_test, output_dir=output_dir)

	Util.exec_time(start, "Running")

if __name__ == '__main__':
	main()
