"""
Forward propagation utilities.

These functions implement the mathematical forward pass operations
used in neural network layers. They serve as the pure-functional
reference implementation, mirroring what the Layer classes do internally.
"""
import numpy as np
from utils.activation import get_activation


def linear_forward(X, W, b):
    """
    Compute the pre-activation (linear transformation) of a layer.

    Args:
        X : input array, shape (n_samples, input_dim)
        W : weight matrix, shape (input_dim, output_dim)
        b : bias vector, shape (1, output_dim)

    Returns:
        Z : pre-activation output, shape (n_samples, output_dim)
    """
    return np.dot(X, W) + b


def activation_forward(Z, activation_name):
    """
    Apply an activation function to the pre-activation Z.

    Args:
        Z               : pre-activation array
        activation_name : string key (e.g. 'relu', 'sigmoid')

    Returns:
        A : post-activation array
    """
    act_fn, _ = get_activation(activation_name)
    return act_fn(Z)


def forward_pass(X, weights, biases, activations):
    """
    Run a full forward pass through all layers.

    Args:
        X           : input data, shape (n_samples, n_features)
        weights     : list of weight matrices W per layer
        biases      : list of bias vectors b per layer
        activations : list of activation name strings per layer

    Returns:
        outputs : list of post-activation arrays A for each layer
        zs      : list of pre-activation arrays Z for each layer
    """
    assert len(weights) == len(biases) == len(activations)

    outputs = []   # A values (post-activation)
    zs = []        # Z values (pre-activation)

    A = X
    for W, b, act in zip(weights, biases, activations):
        Z = linear_forward(A, W, b)
        A = activation_forward(Z, act)
        zs.append(Z)
        outputs.append(A)

    return outputs, zs
