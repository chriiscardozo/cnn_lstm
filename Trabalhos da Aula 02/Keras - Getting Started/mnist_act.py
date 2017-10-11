import keras
from keras.datasets import mnist
from keras import backend as K

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten, Activation, Lambda

############### CUSTOM ACTIVATION ###############
#from keras import backend as K
from keras.utils.generic_utils import get_custom_objects

def custom_activation(x):
    return (K.sigmoid(x) * 5) - 1

get_custom_objects().update({'custom_activation': Activation(custom_activation)})


params = { 'a': 10 }
setattr(K, 'params', params)

def PReLU(x):
	a = K.params['a']
	pos = K.relu(x)
	neg = a * (x - abs(x)) * 0.5
	return pos + neg

from custom import PReLULayer

##################################################


(X_train, y_train), (X_test, y_test) = mnist.load_data()

if K.image_data_format() == 'channels_first':
    X_train = X_train.reshape(X_train.shape[0], 1, 28, 28)
    X_test = X_test.reshape(X_test.shape[0], 1, 28, 28)
else:
    X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
    X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)


X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
print(X_train.shape[0], 'exemplos de treinamento')
print(X_test.shape[0], 'exemplos de testes')

y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

model = Sequential()

model.add(Conv2D(32, kernel_size=(5,5), activation='custom_activation', input_shape=(28,28,1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, kernel_size=(5,5)))
model.add(Lambda(PReLU))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(1024))
# ***
model.add(PReLULayer(name='prelu_param'))
# ***
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

model.compile(	loss='categorical_crossentropy',
				optimizer='adam',
				metrics=['accuracy'])

model.fit(	X_train, y_train, batch_size=1000, epochs=1,verbose=1,
			validation_data=(X_test, y_test))

score = model.evaluate(X_test, y_test, verbose=0)
print('Acurácia:', score[1])
