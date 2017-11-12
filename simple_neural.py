from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout
from keras import regularizers
from keras.utils import np_utils
from keras.optimizers import Adam
import numpy as np
from keras.datasets import mnist
import h5py
from quiver_engine import server

(train_x, train_y), (test_x, test_y) = mnist.load_data()

train_x = train_x.reshape(60000, 784)
test_x = test_x.reshape(10000, 784)
train_x = train_x.astype('float32')
test_x = test_x.astype('float32')

train_x /=255
test_x /=255

train_y = np_utils.to_categorical(train_y, 10)
test_y = np_utils.to_categorical(test_y, 10)

'''def neural_network():
    model = Sequential()

    model.add(Dense(128, input_shape=(784,), activation='relu', kernel_regularizer = regularizers.l2(0.01)))
    model.add(Dropout(0.3))
    model.add(Dense(128,activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(10, activation='softmax'))

    model.summary()

    model.compile(loss='categorical_crossentropy', optimizer = Adam(), metrics=['accuracy'])

    model.fit(train_x, train_y, batch_size = 2000, epochs = 20, validation_split = 0.2)

    model.save('simple_model.h5')

    return model

model = neural_network()'''

model = load_model('simple_model.h5')

server.launch(model)

#score = model.evaluate(test_x, test_y, verbose=1)
#print(score[0],score[1])
