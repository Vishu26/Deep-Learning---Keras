from keras.layers import Dense, Flatten
from keras.models import Sequential, load_model
from keras.utils import np_utils
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
import h5py

class LeNet:
    def build(input_shape, classes):
        model = Sequential()
        model.add(Conv2D(20, kernel_size = 5, padding='same', input_shape=input_shape, activation='relu'))
        model.add(MaxPooling2D(pool_size = (2,2), strides = (2,2), dim_ordering="th"))
        model.add(Conv2D(50, kernel_size = 5, border_mode = 'same', activation='relu'))
        model.add(MaxPooling2D(pool_size = (2,2), strides = (2,2), dim_ordering="th"))
        model.add(Flatten())
        model.add(Dense(500, activation = 'relu'))
        model.add(Dense(classes, activation = 'softmax'))
        return model

INPUT_SHAPE = (1, 28, 28)

(train_x, train_y), (test_x, test_y) = mnist.load_data()

train_x = train_x.astype('float32')
test_x = test_x.astype('float32')

train_x /=255
test_x /=255

train_x = train_x[:, np.newaxis, :, :]
test_x = test_x[:, np.newaxis, :, :]

train_y = np_utils.to_categorical(train_y, 10)
test_y = np_utils.to_categorical(test_y, 10)

'''model = LeNet.build(input_shape = INPUT_SHAPE, classes = 10)
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
model.summary()
model.fit(train_x, train_y, epochs=20, batch_size = 512, validation_split = 0.2)
model.save('convNet.h5')'''

model = load_model('convNet.h5')
score = model.evaluate(test_x, test_y, verbose = 1)
print(score[0], score[1])
