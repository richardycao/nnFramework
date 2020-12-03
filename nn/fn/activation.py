import numpy as np

# Activation functions
def sigmoid(x):
  return 1/(1+np.exp(-x))
def relu(x):
  return np.maximum(x,0)
def tanh(x):
  return np.tanh(x)
def softmax(x):
  """Compute the softmax of vector x in a numerically stable way."""
  shiftx = x - np.max(x)
  exps = np.exp(shiftx)
  return exps / np.sum(exps)

# Activation function gradients
def d_sigmoid(x):
  return sigmoid(x)*(1-sigmoid(x))
def d_relu(x):
  return np.maximum(np.sign(x),0)
def d_tanh(x):
  return 1 - np.power(np.tanh(x),2)
def d_softmax(x):
  pass # TODO