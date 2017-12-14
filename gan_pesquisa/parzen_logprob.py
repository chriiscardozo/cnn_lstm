import sys, os, csv
import numpy as np
import Util
from sklearn.neighbors.kde import KernelDensity
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from Util import find_csv_filenames

def log_proba(X_test, folder, file_name):
	with open(os.path.join(folder, file_name), 'r') as f:
		reader = csv.reader(f, delimiter=',')
		samples = []
		for row in reader: samples.append(row)
		samples = np.array(samples)

		# Cross-validation to find best bandwidth
		params = {'bandwidth': np.linspace(0, 1, 10)}
		grid = GridSearchCV(KernelDensity(kernel='gaussian'), params, n_jobs=8)
		grid.fit(samples)
		print("best bandwidth: {0}".format(grid.best_estimator_.bandwidth))
		kde = grid.best_estimator_

		scores = kde.score_samples(X_test)
		return [np.mean(scores), np.std(scores)] # return mean log prob and std log prob

def main():
	if(len(sys.argv) < 2):
		print("Error: missing input folder parameter")
		exit(0)
	folder = sys.argv[1]

	files = find_csv_filenames(folder)
	files = sorted(files, key=lambda x: int(x.split('.')[0].split('_')[1]))

	X_train, y_train, X_test, y_test = Util.load_mnist_data()

	x = []
	lls_avg = []
	lls_std = []

	for file_name in files:
		print("Calculating for file", file_name)
		x.append(int(file_name.split('.')[0].split('_')[1]))
		avg, std = log_proba(X_test, folder, file_name)
		lls_avg.append(avg)
		lls_std.append(std)
		print("avg:", avg, " | std:", std)

	save_results(folder, x, lls_avg, lls_std)

if __name__ == '__main__':
	main()