import sys
import os
from Parzen import ParsenDensityEstimator as Parzen
import tensorflow as tf
import Util
import csv
import numpy as np

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

def gpu_loglikelihood(session):
	p = Parzen()

def main():
	session = None
	outputs = Util.find_output_folders()

	for output_folder in outputs[0:1]:
		print("** output folder", output_folder, "***")

		execution_folders = [x for x in sorted(os.listdir(output_folder)) if x.isdigit()]

		all_x = []
		all_times = []
		all_d_losses = []
		all_g_losses = []
		all_d_accuracies = []

		for execution_folder in execution_folders:
			print(int(execution_folder) + 1, "of", len(execution_folders))
			
			current_folder = os.path.join(output_folder, execution_folder)
			
			x , times, d_losses, g_losses, d_accuracies = generate_general_graphics(current_folder)
			all_x.append(x)
			all_times.append(times)
			all_d_losses.append(d_losses)
			all_g_losses.append(g_losses)
			all_d_accuracies.append(d_accuracies)
			
			if('cpu' in sys.argv):
				cpu_loglikelihood()
			else: # default calculate parzen using gpu
				if(session is None):
					session = tf.Session()
				gpu_loglikelihood(session)

		all_x = np.array(all_x).mean(axis=0)
		all_times = np.array(all_times).mean(axis=0)
		all_d_losses = np.array(all_d_losses).mean(axis=0)
		all_g_losses = np.array(all_g_losses).mean(axis=0)
		all_d_accuracies = np.array(all_d_accuracies).mean(axis=0)

		Util.generate_graphics(all_x, all_times, all_d_losses, all_g_losses, all_d_accuracies, output_folder)

if __name__ == '__main__':
	main()