from nn.models import Sequential
from nn.layers import Dense
import numpy as np

n = 10
X = np.random.normal(size=(n,1))
y = (X>0)*1

model = Sequential()
model.add(Dense(2, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='sgd')
model.fit(X, y, epochs=50, alpha=0.1)

