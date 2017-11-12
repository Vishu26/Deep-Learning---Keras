from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.utils import np_utils
import numpy as np
import h5py

np.random.seed(7)

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)


x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

def conv(input_shape, classes):
    model = Sequential()
    model.add(Conv2D(32, (2,2), padding='same', activation='relu', input_shape=input_shape))
    model.add(Conv2D(32, (2,2), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(64, (2,2), padding='same', activation='relu'))
    model.add(Conv2D(64, (2,2), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(128, (2,2), padding='same', activation='relu'))
    model.add(Conv2D(128, (2,2), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(classes, activation='softmax'))

    return model

model = conv((32, 32, 3), 10)
model.summary()
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=10, batch_size= 512, validation_split=0.2)

model.save('cifarr.h5')
