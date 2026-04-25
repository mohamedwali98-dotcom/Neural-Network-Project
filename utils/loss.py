"""
Loss (cost) functions and their gradients with respect to the output layer.
"""
import numpy as np


def mse(y_pred, y_true):
    """Mean Squared Error — used for regression."""
    return np.mean((y_pred - y_true) ** 2)


def mse_derivative(y_pred, y_true):
    """dL/dy_pred for MSE"""
    n = y_pred.shape[0]
    return 2 * (y_pred - y_true) / n


def binary_cross_entropy(y_pred, y_true):
    """Binary Cross-Entropy — used for binary classification."""
    eps = 1e-15
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


def binary_cross_entropy_derivative(y_pred, y_true):
    """dL/dy_pred for BCE"""
    eps = 1e-15
    y_pred = np.clip(y_pred, eps, 1 - eps)
    n = y_pred.shape[0]
    return (-(y_true / y_pred) + (1 - y_true) / (1 - y_pred)) / n


def categorical_cross_entropy(y_pred, y_true):
    """
    Categorical Cross-Entropy — used for multi-class classification.
    y_true: one-hot encoded or class indices.
    """
    eps = 1e-15
    y_pred = np.clip(y_pred, eps, 1 - eps)
    if y_true.ndim == 1:
        # class indices — convert to one-hot
        n = y_pred.shape[0]
        y_oh = np.zeros_like(y_pred)
        y_oh[np.arange(n), y_true.astype(int)] = 1
        y_true = y_oh
    return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))


def categorical_cross_entropy_derivative(y_pred, y_true):
    """
    Combined gradient of softmax + categorical CE = y_pred - y_true.
    This is used directly instead of separating softmax derivative.
    """
    eps = 1e-15
    y_pred = np.clip(y_pred, eps, 1 - eps)
    n = y_pred.shape[0]
    if y_true.ndim == 1:
        y_oh = np.zeros_like(y_pred)
        y_oh[np.arange(n), y_true.astype(int)] = 1
        y_true = y_oh
    return (y_pred - y_true) / n


LOSSES = {
    "mse": (mse, mse_derivative),
    "binary_cross_entropy": (binary_cross_entropy, binary_cross_entropy_derivative),
    "categorical_cross_entropy": (categorical_cross_entropy, categorical_cross_entropy_derivative),
}


def get_loss(name):
    """Return (loss_fn, loss_derivative_fn) by name."""
    name = name.lower()
    if name not in LOSSES:
        raise ValueError(f"Unknown loss '{name}'. Choose from: {list(LOSSES.keys())}")
    return LOSSES[name]
