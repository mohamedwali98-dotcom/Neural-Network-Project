"""
Activation functions and their derivatives.
All functions operate on NumPy arrays.
"""
import numpy as np


def sigmoid(z):
    """Sigmoid activation: 1 / (1 + exp(-z))"""
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_derivative(z):
    """Derivative of sigmoid w.r.t. z (when input is pre-activation)"""
    s = sigmoid(z)
    return s * (1 - s)


def relu(z):
    """ReLU activation: max(0, z)"""
    return np.maximum(0, z)


def relu_derivative(z):
    """Derivative of ReLU w.r.t. z"""
    return (z > 0).astype(float)


def tanh(z):
    """Tanh activation"""
    return np.tanh(z)


def tanh_derivative(z):
    """Derivative of tanh w.r.t. z"""
    return 1.0 - np.tanh(z) ** 2


def softmax(z):
    """Softmax activation (numerically stable)"""
    z_shifted = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z_shifted)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


def softmax_derivative(z):
    """
    For softmax with categorical cross-entropy, the combined gradient simplifies.
    This returns ones (the actual gradient is computed in the loss layer).
    """
    return np.ones_like(z)


def step(z):
    """Historical perceptron step activation"""
    return (z >= 0).astype(float)


def step_derivative(z):
    """Step function has zero derivative everywhere (use 1 as a placeholder)"""
    return np.ones_like(z)


ACTIVATIONS = {
    "sigmoid": (sigmoid, sigmoid_derivative),
    "relu": (relu, relu_derivative),
    "tanh": (tanh, tanh_derivative),
    "softmax": (softmax, softmax_derivative),
    "step": (step, step_derivative),
    "linear": (lambda z: z, lambda z: np.ones_like(z)),
}


def get_activation(name):
    """Return (activation_fn, derivative_fn) tuple by name."""
    name = name.lower()
    if name not in ACTIVATIONS:
        raise ValueError(f"Unknown activation '{name}'. Choose from: {list(ACTIVATIONS.keys())}")
    return ACTIVATIONS[name]
