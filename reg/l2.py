"""
L2 (weight decay) regularization.
"""
import numpy as np


def l2_penalty(weights_list, lambda_):
    """
    Compute the L2 regularization penalty term.
    weights_list: list of weight matrices (excluding biases).
    lambda_: regularization strength.
    """
    penalty = 0.0
    for W in weights_list:
        penalty += np.sum(W ** 2)
    return 0.5 * lambda_ * penalty


def l2_gradient(W, lambda_):
    """
    Gradient of the L2 penalty w.r.t. weight matrix W.
    Returns lambda_ * W to be added to the weight gradient.
    """
    return lambda_ * W
