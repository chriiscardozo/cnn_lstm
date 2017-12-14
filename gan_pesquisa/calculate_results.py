import sys
import os
from Parzen import ParsenDensityEstimator as Parzen
import tensorflow as tf
import Util
import csv
import numpy as np
import time

def generate_general_graphics(folder):
	files = ['x', 'times', 'd_losses', 'g_losses', 'd_accuracies']
	infos = []
	for name in files:
		with open(os.path.join(folder, name + ".csv"), 'r') as f:
			reader = csv.reader(f, delimiter=',')
			infos.append([float(x) for x in next(reader)])

	x , times, d_losses, g_losses, d_accuracies = infos
	Util.generate_graphics(x, times, d_losses, g_losses, d_accuracies, folder)

	return infos

def gpu_parzen(session, X_test, folder, filename):
	p = Parzen()
	with open(os.path.join(folder, filename), 'r') as f:
		reader = csv.reader(f, delimiter=',')
		samples = []
		for row in reader: samples.append([float(x) for x in row])
		samples = np.array(samples)

		lls_avg_sigma = []
		lls_std_sigma = []
		# Cross-validation to find best bandwidth
		params = {'bandwidth': np.linspace(0.1, 1, 10)}
		for sigma in params['bandwidth']:
			ll_avg, ll_std = p.get_ll(X_test, samples, sigma, session,batch_size=1000)
			lls_avg_sigma.append(ll_avg)
			lls_std_sigma.append(ll_std)

		index = np.array(lls_avg_sigma).argmax()
		print("best sigma found:", params['bandwidth'][index])
		return [lls_avg_sigma[index], lls_std_sigma[index]]


def gpu_loglikelihood(X_test, session, folder):
	lls_mean = []
	lls_std = []

	filenames = Util.find_csv_filenames(folder,sorted_fnt=lambda x: int(x.split('.')[0].split('_')[1]))

	for index, filename in enumerate(filenames):
		start = time.time()

		mean, std = gpu_parzen(session, X_test, folder, filename)
		lls_mean.append(mean)
		lls_std.append(std)

		print("Sample", index,"| loop time:", time.time()-start, "seconds")

	return [lls_mean, lls_std]

def main():
	session = None
	outputs = Util.find_output_folders()
	X_train, y_train, X_test, y_test = Util.load_mnist_data()

	for output_folder in outputs[0:1]:
		execution_folders = [x for x in sorted(os.listdir(output_folder)) if x.isdigit()]

		all_x = []
		all_times = []
		all_d_losses = []
		all_g_losses = []
		all_d_accuracies = []
		all_lls_mean = []
		all_lls_std = []

		for execution_folder in execution_folders:
			print(output_folder, int(execution_folder) + 1, "of", len(execution_folders))
			
			current_folder = os.path.join(output_folder, execution_folder)
			
			x , times, d_losses, g_losses, d_accuracies = generate_general_graphics(current_folder)
			all_x.append(x)
			all_times.append(times)
			all_d_losses.append(d_losses)
			all_g_losses.append(g_losses)
			all_d_accuracies.append(d_accuracies)
			
			lls_mean = []
			lls_std = []

			if('cpu' in sys.argv):
				lls_mean, lls_std = cpu_loglikelihood()
			# default calculatng parzen is using gpu
			else:
				if(session is None): session = tf.Session()
				lls_mean, lls_std = gpu_loglikelihood(X_test, session, current_folder)

			all_lls_mean.append(lls_mean)
			all_lls_std.append(lls_std)
			Util.save_ll_results(current_folder, x, lls_mean, lls_std)

		all_x = np.array(all_x).mean(axis=0)
		all_times = np.array(all_times).mean(axis=0)
		all_d_losses = np.array(all_d_losses).mean(axis=0)
		all_g_losses = np.array(all_g_losses).mean(axis=0)
		all_d_accuracies = np.array(all_d_accuracies).mean(axis=0)
		all_lls_mean = np.array(all_lls_mean).mean(axis=0)
		all_lls_std = np.array(all_lls_std).mean(axis=0)

		Util.generate_graphics(all_x, all_times, all_d_losses, all_g_losses, all_d_accuracies, output_folder)
		Util.save_ll_results(output_folder, all_x, all_lls_mean, all_lls_std)

if __name__ == '__main__':
	main()