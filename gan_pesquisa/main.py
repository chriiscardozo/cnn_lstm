from GAN import *
import Util
import numpy as np

def main():
	np.random.seed(42)
	X_train, y_train, X_test, y_test = Util.load_mnist_data()

	output_dir = 'output'
	Util.reset_dir(output_dir)

	gan = GAN()
	gan.train(X_train, epochs=1000, batch_size=128, verbose_step=250, output_dir=output_dir)


if __name__ == '__main__':
	main()