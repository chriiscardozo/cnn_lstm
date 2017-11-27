import numpy as np
from keras.datasets import mnist
import matplotlib.pyplot as plt
import os, shutil
import time

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

def generate_graphics(times, d_lossses, g_losses, output_dir):
	plt.close('all')
	x = np.linspace(0, len(times), len(times))

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

def plot_generated_images(e, generatedImages, output_dir, dim=(5, 5), figsize=(5, 5)):
	plt.close('all')

	plt.figure(figsize=figsize)
	for i in range(generatedImages.shape[0]):
		plt.subplot(dim[0], dim[1], i+1)
		plt.imshow(generatedImages[i], interpolation='nearest', cmap='gray_r')
		plt.axis('off')
	plt.tight_layout()
	plt.savefig(os.path.join(output_dir, str(e) + '.png'))

def reset_dir(folder):
	if os.path.exists(folder):
		shutil.rmtree(folder)
	os.makedirs(folder)

def exec_time(start, msg):
	end = time.time()
	delta = end - start
	if(delta > 60): print("Tempo: " + str(delta/60.0) + " min [" + msg + "]")
	else: print("Tempo: " + str(int(delta)) + " s [" + msg + "]")