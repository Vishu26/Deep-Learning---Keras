from keras.datasets import mnist
from keras.optimizers import SGD
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Activation, Dense
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.utils import np_utils
from keras.layers import Dropout
import h5py

np.random.seed(7)
(train_x, train_y), (test_x, test_y) = mnist.load_data()
'''plt.subplot(221)
plt.imshow(train_x[0], cmap=plt.get_cmap('gray'))
plt.subplot(222)
plt.imshow(train_x[1], cmap=plt.get_cmap('gray'))
plt.subplot(223)
plt.imshow(train_x[2], cmap=plt.get_cmap('gray'))
plt.subplot(224)
plt.imshow(train_x[3], cmap=plt.get_cmap('gray'))
# show the plot
plt.show()'''

num_pixels = train_x.shape[1] * train_x.shape[2]
train_x = train_x.reshape(train_x.shape[0], num_pixels).astype('float32')
test_x = test_x.reshape(test_x.shape[0], num_pixels).astype('float32')

train_x = train_x / 255
train_x = train_x / 255

train_y = np_utils.to_categorical(train_y)
test_y = np_utils.to_categorical(test_y)
num_classes = test_y.shape[1]
'''
# define baseline model
def baseline_model():
    # create model
    model = Sequential()
    model.add(Dense(num_pixels, input_dim=num_pixels, kernel_initializer='normal', activation='relu'))
    model.add(Dense(num_classes, kernel_initializer='normal', activation='softmax'))
    #sgd = SGD(lr=0.01)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# build the model
model = baseline_model()
# Fit the model
model.fit(train_x, train_y, validation_data=(test_x, test_y), epochs=20, batch_size=50, verbose=2)
# Final evaluation of the model
scores = model.evaluate(test_x, test_y, verbose=0)
print("Baseline Error: %.2f%%" % (100-scores[1]*100))

model.save('my_model.h5')
'''

model = load_model('my_model.h5')

result = model.predict(test_x)
acc = 0


'''for i in range(5000):
    fl = 0
    for j in range(10):
        if result[i][j]!=test_y[i][j]:
            fl = 1
            break
    if fl==0:
        acc+=1
print(acc/50)'''

'''plt.plot(list(range(200)), [np.where(x==1)[0][0] for x in result[0:200]])
plt.plot(list(range(200)), [np.where(x==1)[0][0] for x in test_y[0:200]],color='r')
plt.yticks(list(range(10)))'''

plt.show()
