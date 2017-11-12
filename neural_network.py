from keras.models import Sequential
from keras.layers import Dense
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

'''np.random.seed(0)

data = input_data.read_data_sets('/',one_hot=True)

print(data.shape)'''

df = np.loadtxt("pima-indians-diabetes.csv",delimiter=',')

x = df[:,0:8]

y = df[:,8]

print(x)
model = Sequential()
model.add(Dense(12,input_dim=8,activation='relu'))
model.add(Dense(8,activation='relu'))
model.add(Dense(1,activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(x,y,epochs=1000,batch_size=40)

scores = model.evaluate(x,y)

print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
