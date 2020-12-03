import numpy as np

# Loss functions
def binary_crossentropy(y_hat, y): # for sigmoid
  return ((y*np.log(y_hat) if y_hat != 0 else 0) + ((1-y)*np.log(1-y_hat)) if y_hat != 1 else 0)
def crossentropy(y_hat, y): # for softmax
  return np.sum(-y * np.log(y_hat))

# Loss function gradients
def d_binary_crossentropy(y_hat, y): # for sigmoid
  return -((np.divide(y, y_hat) if y_hat != 0 else 0) - (np.divide(1 - y, 1 - y_hat) if y_hat != 1 else 0))
  #return y_hat - y
def d_crossentropy(y_hat, y): # for softmax
  return y_hat - y