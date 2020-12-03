import numpy as np
from .layer import Layer

class Dropout(Layer):
  def __init__(self, ratio=0.5):
    super().__init__()
    self.ratio = ratio

  def setup(self, input_size):
    self.input_size = input_size
    return input_size

  def forward(self, x):
    self.drop_vec = np.random.choice([0,1], size=x.shape, p=[self.ratio,1-self.ratio])
    a = np.multiply(x, self.drop_vec)
    return a

  def backward(self, dA, optimizer, alpha):
    return dA

  def predict(self, x):
    return x