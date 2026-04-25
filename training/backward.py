"""
Backpropagation utilities.

These functions implement the chain-rule gradient computation
used during training. They serve as the pure-functional reference
implementation, mirroring what the MLP model does internally.
"""
import numpy as np
from utils.activation import get_activation


def activation_backward(dA, Z, activation_name):
    """
    Compute gradient of loss with respect to pre-activation Z.

    Uses the chain rule:  dL/dZ = dL/dA * dA/dZ

    Args:
        dA              : gradient of loss w.r.t. post-activation A
        Z               : pre-activation values (cached from forward pass)
        activation_name : activation function name string

    Returns:
        dZ : gradient of loss w.r.t. pre-activation Z
    """
    _, act_deriv = get_activation(activation_name)
    return dA * act_deriv(Z)


def linear_backward(dZ, X, W):
    """
    Compute gradients of loss with respect to W, b, and the input X.

    Args:
        dZ : gradient of loss w.r.t. pre-activation Z, shape (n, out)
        X  : input to this layer (cached from forward pass), shape (n, in)
        W  : weight matrix, shape (in, out)

    Returns:
        dW : gradient w.r.t. weights,  shape (in, out)
        db : gradient w.r.t. biases,   shape (1, out)
        dX : gradient w.r.t. input X,  shape (n, in)
    """
    n = X.shape[0]
    dW = np.dot(X.T, dZ)          # (in, out)
    db = np.sum(dZ, axis=0, keepdims=True)  # (1, out)
    dX = np.dot(dZ, W.T)          # (n, in)
    return dW, db, dX


def backward_pass(dA_last, zs, outputs, X, weights, activations):
    """
    Full backpropagation through all layers.

    Args:
        dA_last     : gradient of loss w.r.t. the final layer output
        zs          : list of pre-activation arrays Z (from forward_pass)
        outputs     : list of post-activation arrays A (from forward_pass)
        X           : original input data
        weights     : list of weight matrices per layer
        activations : list of activation name strings per layer

    Returns:
        grads : list of dicts, each with 'dW', 'db' for the corresponding layer.
                Index 0 = first (input) layer, last = output layer.
    """
    n_layers = len(weights)
    grads = [None] * n_layers

    dA = dA_last
    for i in reversed(range(n_layers)):
        Z = zs[i]
        # Input to this layer: X if first layer, else previous layer output
        A_prev = outputs[i - 1] if i > 0 else X

        # Chain rule through activation
        dZ = activation_backward(dA, Z, activations[i])
        # Chain rule through linear
        dW, db, dA = linear_backward(dZ, A_prev, weights[i])

        grads[i] = {"dW": dW, "db": db}

    return grads
