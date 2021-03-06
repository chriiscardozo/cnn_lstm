import sys
import os
import shutil
from keras.datasets import mnist
import numpy as np
from keras.optimizers import Adam
from keras.initializers import RandomNormal
from keras.models import Model, Sequential
from keras.layers import Reshape, Dense, Dropout, Flatten, Conv2D, LeakyReLU, Activation, Input, concatenate
import matplotlib.pyplot as plt
import time
from sklearn.preprocessing import OneHotEncoder

np.random.seed(42)

def exec_time(start, msg):
	end = time.time()
	delta = end - start
	if(delta > 60): print("Tempo: " + str(delta/60.0) + " min [" + msg + "]")
	else: print("Tempo: " + str(int(delta)) + " s [" + msg + "]")

def generator_model(opt):
	model = Sequential()
	model.add(Dense(256, input_dim=110, kernel_initializer=RandomNormal(stddev=0.02)))
	model.add(LeakyReLU(0.2))
	model.add(Dense(512))
	model.add(LeakyReLU(0.2))
	model.add(Dense(1024))
	model.add(LeakyReLU(0.2))
	model.add(Dense(784))
	model.add(Activation('tanh'))

	model.compile(loss='binary_crossentropy', optimizer=opt)

	return model

def discriminator_model(opt):
	model = Sequential()

	model.add(Dense(1024, input_dim=794, kernel_initializer=RandomNormal(stddev=0.02)))
	model.add(LeakyReLU(0.2))
	model.add(Dropout(0.3))
	model.add(Dense(512))
	model.add(LeakyReLU(0.2))
	model.add(Dropout(0.3))
	model.add(Dense(256))
	model.add(LeakyReLU(0.2))
	model.add(Dropout(0.3))
	model.add(Dense(1, activation='sigmoid'))
	model.compile(loss='binary_crossentropy', optimizer=opt)

	return model

def gan_model(D, G, opt):
	D.trainable = False

	gan_input = Input(shape=(100,))
	label_input = Input(shape=(10,))
	concat_1 = concatenate([gan_input, label_input])
	generator_output = G(concat_1)
	disc_input = concatenate([generator_output, label_input])
	gan_output = D(disc_input)
	gan = Model([gan_input,label_input], gan_output)
	gan.compile(loss='binary_crossentropy', optimizer=opt)

	return gan

def train(X_train, y_train, generator, discriminator, GAN, epochs=23000, verbose_step=250, batch_size=128, output_dir='output'):
	print("*** Training", epochs, "epochs with batch size =", batch_size, "***")
	times = []
	d_lossses = []
	g_losses = []

	start_train = time.time()

	encoder = OneHotEncoder(n_values=10)

	for e in range(epochs+1):
		start = time.time()

		choosed_samples = np.random.randint(0, X_train.shape[0], size=batch_size)

		imageBatch = X_train[choosed_samples]
		labelBatch = encoder.fit_transform([[x] for x in y_train[choosed_samples]]).toarray()

		noise = np.concatenate((np.random.normal(0, 1, size=[batch_size, 100]), labelBatch), axis=1)

		G_images = generator.predict(noise)
		X = np.concatenate([np.concatenate((imageBatch,labelBatch),axis=1) , np.concatenate((G_images, labelBatch),axis=1)])

		y = np.zeros(2*batch_size)
		y[:batch_size] = 0.9

		discriminator.trainable = True
		d_loss = discriminator.train_on_batch(X, y)

		noise = np.random.normal(0, 1, size=[batch_size, 100])
		# noise = np.concatenate((np.random.normal(0, 1, size=[batch_size, 100]), labelBatch), axis=1)
		y = np.ones(batch_size)
		discriminator.trainable = False
		g_loss = GAN.train_on_batch([noise, labelBatch], y)

		d_lossses.append(d_loss)
		g_losses.append(g_loss)
		times.append(time.time() - start)

		if(e % verbose_step == 0):
			print(str(e) + ": d_loss =", d_loss, "| g_loss =", g_loss)
			
			for num in range(0,10):
				vec = encoder.fit_transform([[num]]).toarray()[0]
				plotGeneratedImages(e, generator, output_dir, vec)

	exec_time(start_train, "Training")
	generate_graphics(times, d_lossses, g_losses, output_dir)

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

def plotGeneratedImages(e, generator, output_dir, one_hot_vec, examples=25, dim=(5, 5), figsize=(10, 10)):
    noise = np.concatenate([np.random.normal(0, 1, size=[examples, 100]), [one_hot_vec]*examples ], axis=1)

    generatedImages = generator.predict(noise)
    generatedImages = generatedImages.reshape(examples, 28, 28)

    plt.close('all')

    plt.figure(figsize=figsize)
    for i in range(generatedImages.shape[0]):
        plt.subplot(dim[0], dim[1], i+1)
        plt.imshow(generatedImages[i], interpolation='nearest', cmap='gray_r')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, str(e) + '_num' + str(np.where(one_hot_vec == 1)[0]) + '.png'))

def main():
	if(len(sys.argv) > 1):
		folder = 'output_'+sys.argv[1]
	else:
		folder = 'output'
	if os.path.exists(folder):
		shutil.rmtree(folder)
	os.makedirs(folder)

	(X_train, y_train), (X_test, y_test) = mnist.load_data()
	X_train = (X_train.astype(np.float32)-127.5)/127.5
	X_train = X_train.reshape(60000, 784)

	opt = Adam(lr=0.0002, beta_1=0.5)

	generator = generator_model(opt)
	discriminator = discriminator_model(opt)
	GAN = gan_model(discriminator, generator, opt)

	train(X_train, y_train, generator, discriminator, GAN, output_dir=folder)

if __name__ == '__main__':
	start = time.time()
	main()
	exec_time(start, "All")
