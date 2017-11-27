from GAN import *
import Util
import numpy as np
import time

def main():
	start = time.time()

	np.random.seed(42)
	X_train, y_train, X_test, y_test = Util.load_mnist_data()

	output_dir = 'output'
	Util.reset_dir(output_dir)

	gan = GAN()
	gan.train(X_train, output_dir=output_dir)

	Util.exec_time(start, "Running")


if __name__ == '__main__':
	main()