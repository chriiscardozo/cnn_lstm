# from GAN import *
import Models.GAN_default as GAN_default
import Models.GAN_Gen_BHAA as GAN_Gen_BHAA
import Models.GAN_Gen_BHSA as GAN_Gen_BHSA
import numpy as np
import time
import sys
import Util

def main():
	start = time.time()

	np.random.seed(42)
	X_train, y_train, X_test, y_test = Util.load_mnist_data()

	models = { "gan_default": GAN_default,
			   "gan_gen_bhaa": GAN_Gen_BHAA,
			   "gan_gen_bhsa": GAN_Gen_BHSA }

	if len(sys.argv) < 2:
		print("Error: missing models")

	for model_name in sys.argv[1:]:
		if model_name in models:
			print("Running model:", model_name)
			output_dir = 'output_' + model_name
			Util.reset_dir(output_dir)
			gan = models[model_name].GAN()
			gan.train(X_train, X_test, output_dir=output_dir)

	Util.exec_time(start, "Running")

if __name__ == '__main__':
	main()
