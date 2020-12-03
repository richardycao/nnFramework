from ..fn import *

class Layer(object):
  def setup(self, input_size):
    return input_size

  def forward(self, x):
    return 0

  def backward(self, dA, optimizer, alpha):
    return 0

  def predict(self, x):
    return 0

  def set_params(self, activation='sigmoid'):
    if activation == 'sigmoid':
      self.activation = sigmoid
      self.d_activation = d_sigmoid
    elif activation == 'relu':
      self.activation = relu
      self.d_activation = d_relu
    elif activation == 'tanh':
      self.activation = tanh
      self.d_activation = d_tanh
    elif activation == 'softmax':
      self.activation = softmax
      self.d_activation = None # TODO
    else:
      raise Exception('Non-supported activation function')