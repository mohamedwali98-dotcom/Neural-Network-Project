"""
Dropout regularization.
"""
import numpy as np


class Dropout:
    """
    Inverted dropout layer.
    During training: randomly zero out units with probability `rate`, scale up remaining.
    During inference: pass through unchanged.
    """

    def __init__(self, rate=0.5):
        """
        rate: fraction of units to DROP (set to 0). e.g. rate=0.2 keeps 80%.
        """
        assert 0 <= rate < 1, "Dropout rate must be in [0, 1)"
        self.rate = rate
        self.mask = None

    def forward(self, X, training=True):
        if not training or self.rate == 0:
            return X
        # Inverted dropout: scale by 1/(1-rate) to preserve expected value
        self.mask = (np.random.rand(*X.shape) > self.rate) / (1.0 - self.rate)
        return X * self.mask

    def backward(self, dout):
        """Pass gradient through the mask."""
        if self.mask is None:
            return dout
        return dout * self.mask
