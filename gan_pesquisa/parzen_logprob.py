import sys, os, csv
import numpy as np
import Util
from sklearn.neighbors.kde import KernelDensity
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from KernelDensity import KernelDensity as KDE

def find_csv_filenames(path_to_dir, prefix="samples_", suffix=".csv"):
    filenames = os.listdir(path_to_dir)
    return [ filename for filename in filenames if filename.startswith( prefix ) and filename.endswith( suffix ) ]

def save_results(folder, x, lls_avg, lls_std):
	with open(os.path.join(folder, 'lls_avg.csv'), 'w') as f:
		writer = csv.writer(f, delimiter=',')
		writer.writerow(lls_avg)
	with open(os.path.join(folder, 'lls_std.csv'), 'w') as f:
		writer = csv.writer(f, delimiter=',')
		writer.writerow(lls_std)

	plt.clf()
	plt.title("GAN MNIST - Negative Log-Likelihood per epoch (log-scalar scale)")
	plt.ylabel('Negative Log-Likelihood')
	plt.xlabel('epoch')
	plt.yscale('log')
	plt.text(max(x)/2,-min(lls_avg)/10,'min = ' + str(-int(max(lls_avg))) + '\nepoch = ' + str(x[np.array(lls_avg).argmax()]))
	plt.plot(x, [-float(x) for x in lls_avg])
	plt.savefig(os.path.join(folder, 'll.png'))
	# plt.show()

def log_proba(X_test, folder, file_name, session):
	with open(os.path.join(folder, file_name), 'r') as f:
		reader = csv.reader(f, delimiter=',')
		samples = []
		for row in reader: samples.append(row)
		samples = np.array(samples)

		bands = np.logspace(-1, 0, 10)
		lls_bands = []
		for b in bands:
			kde = KDE(samples, b)
			lls = session.run(kde._log_prob(X_test))

			lls_bands.append( lls )

		tmp = np.array([x.mean() for x in lls_bands])
		print(tmp)
		argmax = tmp.argmax()
		bandwidth = bands[argmax]
		print("best bandwidth:", bandwidth)

		return [np.mean(lls_bands[argmax]), np.std(lls_bands[argmax])] # return mean log prob and std log prob

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
