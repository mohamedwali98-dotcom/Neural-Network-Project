"""
Neural network layers for building the MLP and Perceptron.
"""
import numpy as np


class Dense:
    """A fully connected (dense) layer."""
    
    def __init__(self, input_size, output_size):
        # He initialization for variance control
        self.weights = np.random.randn(input_size, output_size) * np.sqrt(2.0 / input_size)
        self.bias = np.zeros((1, output_size))
        self.x = None
        self.dweights = None
        self.dbias = None

    def forward(self, x, training=True):
        self.x = x
        return np.dot(x, self.weights) + self.bias

    def backward(self, dout):
        """
        dout: gradient of loss wrt output of this layer
        Returns gradient of loss wrt input of this layer
        """
        # Gradients wrt parameters
        self.dweights = np.dot(self.x.T, dout)
        self.dbias = np.sum(dout, axis=0, keepdims=True)
        # Gradient wrt input
        return np.dot(dout, self.weights.T)


class ActivationLayer:
    """A generic activation layer."""
    
    def __init__(self, activation_fn, derivative_fn):
        self.activation_fn = activation_fn
        self.derivative_fn = derivative_fn
        self.x = None

    def forward(self, x, training=True):
        self.x = x
        return self.activation_fn(x)

    def backward(self, dout):
        return dout * self.derivative_fn(self.x)

