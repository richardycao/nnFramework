from nn.models import Sequential
from nn.layers import Dense, Recurrent, Dropout
import numpy as np

def DenseTest():
  n = 100
  X = np.random.normal(size=(n,1))
  y = (X>0)*1

  model = Sequential()
  model.add(Dense(2, activation='relu'))
  model.add(Dense(1, activation='sigmoid'))

  model.compile(loss='binary_crossentropy', optimizer='sgd')
  model.fit(X, y, epochs=20, alpha=0.1)

  print("======= Dense Test =======")
  x = np.random.normal(0, 1, size=(1,1))
  print("Input:", x)
  y_hat = model.predict(x)
  print("Prediction:", y_hat)

def RNNTest():
  n = 10
  X = np.arange(0, n).reshape(-1, 1)
  y = (X%2==1)*1

  model = Sequential()
  model.add(Recurrent(1, 3, activation='relu'))
  model.add(Dense(1, activation='sigmoid'))

  model.compile(loss='binary_crossentropy', optimizer='sgd')
  # model.fit(X, y, epochs=20, alpha=0.1)

  print("======= RNN Test =======")
  x = np.random.normal(0, 1, size=(1,1))
  print("Input:", x)
  # y_hat = model.predict(x)
  # print("Prediction:", y_hat)

# DenseTest()
RNNTest()