import numpy as np
from keras.datasets import mnist
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import os, shutil
import time
import tensorflow as tf
import csv

def convert_to_one_hot_vector(y, n_classes):
	targets = np.array(y).reshape(-1)
	one_hot_targets = np.eye(n_classes)[targets]

	return one_hot_targets

def load_mnist_data(one_hot=True):
	print("*** Loading MNIST dataset ***")

	(X_train, y_train), (X_test, y_test) = mnist.load_data()

	X_train = (X_train.astype(np.float32)-127.5)/127.5
	X_train = X_train.reshape(60000, 784)

	X_test = (X_test.astype(np.float32)-127.5)/127.5
	X_test = X_test.reshape(10000, 784)

	if(one_hot):
		y_train = convert_to_one_hot_vector(y_train, 10)
		y_test = convert_to_one_hot_vector(y_test, 10)


	return [X_train, y_train, X_test, y_test]

def generate_graphics(x, times, d_lossses, g_losses, d_accuracies, output_dir):
	plt.close('all')

	plt.clf()
	plt.title("GAN MNIST - Exec time per epoch")
	plt.ylabel('seconds')
	plt.xlabel('epoch')
	plt.plot(x[1:], times[1:])
	plt.savefig(os.path.join(output_dir, 'times.png'))
	# plt.show()

	plt.clf()
	plt.title("GAN MNIST - D and G losses per epoch")
	plt.ylabel('loss(binary crossentropy)')
	plt.xlabel('epoch')
	plt.plot(x, d_lossses, 'b-', label="D loss")
	plt.plot(x, g_losses, 'g-', label="G loss")
	plt.savefig(os.path.join(output_dir, 'losses.png'))
	# plt.show()

	plt.clf()
	plt.title("GAN MNIST - Disciminator accuracy per epoch")
	plt.ylabel('accuracy')
	plt.xlabel('epoch')
	plt.plot(x, d_accuracies)
	plt.savefig(os.path.join(output_dir, 'accuracy.png'))
	# plt.show()

def save_generated_images(e, generator, output_dir, dim=(5, 5), figsize=(5, 5), examples_img=25, examples_csv=1000):
	noise = generator.generate_noise(examples_csv)
	generatedImages = generator.get_model().predict(noise)

	# saving as csv
	with open(os.path.join(output_dir, 'samples_' + str(e) + '.csv'), 'w') as f:
		writer = csv.writer(f, delimiter=',')
		for x in generatedImages:
			writer.writerow(x.tolist())

	generatedImages = generatedImages[np.random.randint(0, generatedImages.shape[0], size=examples_img)]
	generatedImages = generatedImages.reshape(examples_img, 28, 28)

	# saving as img
	plt.close('all')
	plt.figure(figsize=figsize)
	for i in range(generatedImages.shape[0]):
		plt.subplot(dim[0], dim[1], i+1)
		plt.imshow(generatedImages[i], interpolation='nearest', cmap='gray_r')
		plt.axis('off')
	plt.tight_layout()
	plt.savefig(os.path.join(output_dir, str(e) + '.png'))

	return 'samples_' + str(e) + '.csv'

def reset_dir(folder):
	if os.path.exists(folder):
		shutil.rmtree(folder)
	os.makedirs(folder)

def exec_time(start, msg):
	end = time.time()
	delta = end - start
	if(delta > 60): print("Tempo: " + str(delta/60.0) + " min [" + msg + "]")
	else: print("Tempo: " + str(int(delta)) + " s [" + msg + "]")

def save_general_information(values_dict, output_dir):
	for k in values_dict:
		value = values_dict[k]
		with open(os.path.join(output_dir, k + ".csv"), "w") as f:
			writer = csv.writer(f, delimiter=',')
			writer.writerow(value)

def find_output_folders(folder='.', prefix='output_gan_',sorted_fnt=lambda x: x):
	filenames = os.listdir(folder)
	return sorted([ filename for filename in filenames if filename.startswith( prefix ) ], key=sorted_fnt)


def save_ll_results(folder, x, lls_avg, lls_std):
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


def find_csv_filenames(path_to_dir, prefix="samples_", suffix=".csv",sorted_fnt=lambda x: x):
	filenames = os.listdir(path_to_dir)
	return sorted([ filename for filename in filenames if filename.startswith( prefix ) and filename.endswith( suffix ) ], key=sorted_fnt)
