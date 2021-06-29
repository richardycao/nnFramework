import numpy as np
from .layer import Layer

class Recurrent(Layer):
  def __init__(self, length, hidden_size, activation='tanh'):
    super().__init__()
    self.nodes = length
    self.hidden_size = hidden_size
    self.set_params(activation=activation)

  def setup(self, input_size):
    if len(input_size) != 2:
      raise Exception('Expecting input size of (seq_length, d) but got ' + input_size)
    self.input_size = input_size
    self.W_ih = np.random.random(size=(self.hidden_size, self.input_size[1]))
    self.W_hh = np.random.random(size=(self.hidden_size, self.hidden_size))
    self.b = np.random.random(size=(self.hidden_size, 1))

  def forward(self, x):
    x = x.reshape((self.nodes, -1, 1))

    h_t = np.zeros((self.hidden_size, 1))
    for t in range(self.nodes):
      a = np.dot(self.W_hh, h_t) + np.dot(self.W_ih, x[t]) + self.b
      h_t = self.activation(a)

    return h_t

  def backward(self, dO, optimizer, alpha):
    
    pass

  def predict(self, x):
    return self.forward(x)

