from ..fn import *

class Model(object):
  def __init__(self):
    self.layers = []

  def add(self, layer):
    self.layers.append(layer)
  
  def compile(self, loss=None, optimizer=None, metrics=None):
    if loss == 'crossentropy':
      self.loss_function = crossentropy
      self.d_loss_function = d_crossentropy
    elif loss=='binary_crossentropy':
      self.loss_function = binary_crossentropy
      self.d_loss_function = d_binary_crossentropy
    else:
      pass # Don't change loss function

    if optimizer == 'sgd':
      self.optimizer = sgd
    else:
      pass # Don't change optimizer

  def fit(self, X, y, epochs=1, alpha=0.01):
    pass

  def predict(self, X):
    return 0

  def forward(self, x):
    return x

  def backward(self, y_hat, y):
    pass