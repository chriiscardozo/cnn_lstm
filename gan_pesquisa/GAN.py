# GAN model based from: https://github.com/Zackory/Keras-MNIST-GAN

from keras.initializers import RandomNormal
from keras.layers import Input, Dense, LeakyReLU, Activation, Dropout
from keras.models import Sequential, Model
from keras.optimizers import Adam
import Util
import numpy as np
import time

class GAN:
	def __init__(self, optimizer=None, generator=None, discriminator=None):
		if(optimizer is None): optimizer = Adam(lr=0.0002, beta_1=0.5)
		if(generator is None): generator = Generator(optimizer)
		if(discriminator is None): discriminator = Discriminator(optimizer)

		self._generator = generator
		self._discriminator = discriminator
		self._optimizer = optimizer
		self._model = self._create_gan_model()

	def _create_gan_model(self):
		self._discriminator.set_trainable(False)
		
		D = self._discriminator.get_model()
		G = self._generator.get_model()

		gan_input_shape = (self._generator.get_noise_dim(),)
		gan_input = Input(shape=gan_input_shape)
		gan_output = D(G(gan_input))
		model = Model(gan_input, gan_output)
		
		model.compile(loss='binary_crossentropy', optimizer=self._optimizer)

		return model

	def train(self, X_train, epochs=23000, batch_size=128, verbose_step=250, output_dir='output'):
		print("*** Training", epochs, "epochs with batch size =", batch_size, "***")

		times = []
		d_losses = []
		g_losses = []
		ll = [] # TODO: log-likelihood tracking

		start = time.time()

		G_net = self._generator.get_model()
		D_net = self._discriminator.get_model()
		GAN_net = self._model

		for e in range(epochs+1):
			# Discriminator train
			noise = self._generator.generate_noise(batch_size)
			image_batch = X_train[np.random.randint(0, X_train.shape[0], size=batch_size)]
			generated_images = G_net.predict(noise)
			X = np.concatenate([image_batch, generated_images])
			y = np.zeros(2*batch_size)
			y[:batch_size] = 0.9
			self._discriminator.set_trainable(True)
			d_loss = D_net.train_on_batch(X, y)

			# GAN/Generator train
			noise = self._generator.generate_noise(batch_size)
			y = np.ones(batch_size)
			self._discriminator.set_trainable(False)
			g_loss = GAN_net.train_on_batch(noise, y)

			if(e % verbose_step == 0):
				running_time = time.time() - start
				d_losses.append(d_loss)
				g_losses.append(g_loss)
				times.append(running_time)
				start = time.time()
				print(str(e) + ": d_loss =", d_loss, "| g_loss =", g_loss, "| time =", running_time)
				self._plot_images(e, G_net, output_dir)
				
		Util.generate_graphics(times, d_losses, g_losses, output_dir)

	def _plot_images(self, e, G_net, output_dir, examples=25):
		noise = self._generator.generate_noise(examples)
		generatedImages = G_net.predict(noise)
		generatedImages = generatedImages.reshape(examples, 28, 28)

		Util.plot_generated_images(e, generatedImages, output_dir)

class Generator:
	def __init__(self, optimizer, noise_dim=100, output_dim=784):
		self._noise_dim = noise_dim
		self._output_dim = output_dim
		self._optimizer = optimizer
		self._model = self._create_generator()

	def _create_generator(self):
		model = Sequential()

		model.add(Dense(256, input_dim=self._noise_dim, kernel_initializer=RandomNormal(stddev=0.02)))
		model.add(LeakyReLU(0.2))
		model.add(Dense(512))
		model.add(LeakyReLU(0.2))
		model.add(Dense(1024))
		model.add(LeakyReLU(0.2))
		model.add(Dense(self._output_dim))
		model.add(Activation('tanh'))

		model.compile(loss='binary_crossentropy', optimizer=self._optimizer)

		return model

	def generate_noise(self, N):
		return np.random.normal(0, 1, size=[N, self._noise_dim])

	def get_noise_dim(self):
		return self._noise_dim

	def get_model(self):
		return self._model

class Discriminator:
	def __init__(self, optimizer, input_dim=784, output_dim=1):
		self._input_dim = input_dim
		self._output_dim = output_dim
		self._optimizer = optimizer
		self._model = self._create_discriminator()

	def _create_discriminator(self):
		model = Sequential()

		model.add(Dense(1024, input_dim=self._input_dim, kernel_initializer=RandomNormal(stddev=0.02)))
		model.add(LeakyReLU(0.2))
		model.add(Dropout(0.3))
		model.add(Dense(512))
		model.add(LeakyReLU(0.2))
		model.add(Dropout(0.3))
		model.add(Dense(256))
		model.add(LeakyReLU(0.2))
		model.add(Dropout(0.3))
		model.add(Dense(self._output_dim))
		model.add(Activation('sigmoid'))

		model.compile(loss='binary_crossentropy', optimizer=self._optimizer)

		return model

	def set_trainable(self, value):
		self._model.trainable = value

	def get_model(self):
		return self._model