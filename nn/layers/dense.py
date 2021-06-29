import numpy as np
from .layer import Layer

class Dense(Layer):
  def __init__(self, nodes, activation='sigmoid'):
    super().__init__()
    self.nodes = nodes
    self.set_params(activation=activation)

  def setup(self, input_size):
    self.input_size = input_size[0]
    self.w = np.random.random(size=(self.nodes, self.input_size))
    self.b = np.random.random(size=(self.nodes, 1))

    self.x = np.zeros((self.nodes, 1))

    return np.array([self.nodes])

  def forward(self, x):
    x = x.reshape((-1,1))
    self.x = x # save for backprop

    z = np.dot(self.w,x) + self.b
    self.z = z # save for backprop

    a = self.activation(z)
    return a

  def backward(self, dA, optimizer, alpha):           # dA comes from the next layer. For the last layer it's dL/d(y_hat)
    dz = np.multiply(dA, self.d_activation(self.z))   # dz refers to dA/dz
    dw = np.dot(dz, self.x.T)                         # dw refers to dA/dw
    db = dz                                           # db refers to dA/db
    self.w = optimizer(self.w, dw, alpha)
    self.b = optimizer(self.b, db, alpha)

    dX = np.dot(self.w.T, dz)
    
    return dX

  def predict(self, x):
    return self.forward(x)