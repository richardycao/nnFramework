from nn.models import Sequential
from nn.layers import Dense, Dropout
import numpy as np

n = 100
X = np.random.normal(size=(n,1))
y = (X>0)*1

model = Sequential()
model.add(Dense(2, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='sgd')
model.fit(X, y, epochs=20, alpha=0.1)

print("=======")
x = np.random.normal(0, 1, size=(1,1))
print("Input:", x)
y_hat = model.predict(x)
print("Prediction:", y_hat)

