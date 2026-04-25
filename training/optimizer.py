"""
Optimizers for weight updates.
"""
import numpy as np


class SGD:
    """Stochastic Gradient Descent optimizer (can be used for Mini-batch as well)."""
    
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate

    def update(self, layer):
        """
        Updates layer.weights and layer.bias given their gradients.
        """
        if hasattr(layer, 'weights') and layer.dweights is not None:
            layer.weights -= self.learning_rate * layer.dweights
        if hasattr(layer, 'bias') and layer.dbias is not None:
            layer.bias -= self.learning_rate * layer.dbias

