# from GAN import *

import Models.GAN_default as GAN_default
import Models.GAN_Gen_BHAA as GAN_Gen_BHAA
import Models.GAN_Gen_BHSA as GAN_Gen_BHSA
import Models.GAN_Dis_BHAA as GAN_Dis_BHAA
import Models.GAN_Dis_BHSA as GAN_Dis_BHSA
import Models.GAN_Dis_Gen_BHAA as GAN_Dis_Gen_BHAA

import numpy as np
import random
import time
import sys
import Util

def main():
	start = time.time()

	X_train, y_train, X_test, y_test = Util.load_mnist_data()

	models = { "gan_default": GAN_default,
			   "gan_gen_bhaa": GAN_Gen_BHAA,
			   "gan_gen_bhsa": GAN_Gen_BHSA,
			   "gan_dis_bhsa": GAN_Dis_BHSA,
			   "gan_dis_bhaa": GAN_Dis_BHAA,
			   "gan_dis_gen_bhaa": GAN_Dis_Gen_BHAA }

	if len(sys.argv) < 2:
		print("Error: missing models")

	n_executions = 2

	for model_name in sys.argv[1:]:
		if model_name in models:
			output_dir = 'output_' + model_name
			Util.reset_dir(output_dir)

			for i in range(n_executions):
				print("[", model_name, "]", i, "of", n_executions)

				seed = datetime.now()
				random.seed(seed)
				np.random.seed(seed)

				execution_output_dir = os.path.join(output_dir, i)
				Util.reset_dir(execution_output_dir)
				
				gan = models[model_name].GAN()
				gan.train(X_train, X_test, epochs=50000, batch_size=100, verbose_step=500, output_dir=execution_output_dir)

	Util.exec_time(start, "Running")

if __name__ == '__main__':
	main()
