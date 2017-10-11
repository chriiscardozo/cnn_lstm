import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense

(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
print(X_train.shape[0], 'exemplos de treinamento')
print(X_test.shape[0], 'exemplos de testes')

y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

model = Sequential()
model.add(Dense(10, activation='softmax', input_shape=(784,)))

model.summary()

model.compile(	loss='categorical_crossentropy',
				optimizer='sgd',
				metrics=['accuracy'])

model.fit(	X_train, y_train,
			batch_size=100,
			epochs=10,
			verbose=1,
			validation_data=(X_test, y_test))

score = model.evaluate(X_test, y_test)
print('\nAcurácia: ', score[1]) # +- 90.2% de acurácia