from .model import Model
import numpy as np

class Sequential(Model):
  def __init__(self):
    super().__init__()

  def fit(self, X, y, epochs=1, alpha=0.01):
    if X.shape[0] != len(y):
      raise Exception("X and y size mismatch. X:" + str(X.shape[0]) + " and y:" + str(len(y)))

    self.epochs = epochs
    self.alpha = alpha

    input_size = X.shape[1:]
    for layer in self.layers:
      input_size = layer.setup(input_size)

    for e in range(self.epochs):
      loss = 0
      for i in range(X.shape[0]):
        y_hat = self.forward(X[i])
        loss += self.loss_function(y_hat, y[i])
        self.backward(y_hat, y[i])
      print("Loss (epoch "+str(e)+") = "+str(-np.average(loss)))

  def predict(self, x):
    for layer in self.layers:
      #print(x)
      x = layer.predict(x)
    #print(x)
    return x

  def forward(self, x):
    for layer in self.layers:
      x = layer.forward(x)
    return x

  def backward(self, y_hat, y):
    y = y.reshape(y_hat.shape)
    dL = self.d_loss_function(y_hat, y)

    for layer in reversed(self.layers):
      dL = layer.backward(dL, self.optimizer, self.alpha)