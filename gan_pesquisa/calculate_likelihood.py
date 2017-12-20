import sys
import os
import Util
import csv
import numpy as np
import time

def load_statistics(folder):
	files = ['x', 'times', 'd_losses', 'g_losses', 'd_accuracies', 'lls_avg', 'lls_std']
	infos = []
	for name in files:
		with open(os.path.join(folder, name + ".csv"), 'r') as f:
			reader = csv.reader(f, delimiter=',')
			infos.append([float(x) for x in next(reader)])

	x , times, d_losses, g_losses, d_accuracies, lls_avg, lls_std = infos

	return infos


def main():
	outputs = Util.find_output_folders()
	option = 'gpu'
	wait_time = 2*60

	for output_folder in outputs[1:]:
		start = time.time()

		execution_folders = [x for x in sorted(os.listdir(output_folder)) if x.isdigit()]

		all_x = []
		all_times = []
		all_d_losses = []
		all_g_losses = []
		all_d_accuracies = []
		all_lls_mean = []
		all_lls_std = []

		for execution_folder in execution_folders:
			os.system('python3 calculate_likelihood_experiment.py ' + option + ' ' + output_folder + ' ' + execution_folder)

			current_folder = os.path.join(output_folder, execution_folder)
			x , times, d_losses, g_losses, d_accuracies, lls_mean, lls_std  = load_statistics(current_folder)

			all_x.append(x)
			all_times.append(times)
			all_d_losses.append(d_losses)
			all_g_losses.append(g_losses)
			all_d_accuracies.append(d_accuracies)
			all_lls_mean.append(lls_mean)
			all_lls_std.append(lls_std)


			print("Waiting", wait_time, "minutes for temperature reduction")
			time.sleep(wait_time)

		all_x = np.array(all_x).mean(axis=0)
		all_times = np.array(all_times).mean(axis=0)
		all_d_losses = np.array(all_d_losses).mean(axis=0)
		all_g_losses = np.array(all_g_losses).mean(axis=0)
		all_d_accuracies = np.array(all_d_accuracies).mean(axis=0)
		all_lls_mean = np.array(all_lls_mean).mean(axis=0)
		all_lls_std = np.array(all_lls_std).mean(axis=0)

		Util.generate_graphics(all_x, all_times, all_d_losses, all_g_losses, all_d_accuracies, output_folder)
		Util.save_ll_results(output_folder, all_x, all_lls_mean, all_lls_std)
		print("Time to calculate for", output_folder, ">", (time.time() - start)/60.0, "minutes")


if __name__ == '__main__':
	main()