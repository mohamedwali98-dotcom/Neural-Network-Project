"""
Multi-Layer Perceptron (MLP).

Supports:
  - Configurable hidden layers and neurons per layer
  - Any activation function from utils/activation.py
  - L2 regularization
  - Dropout regularization
  - Early stopping
  - Mini-batch gradient descent
  - Binary classification, multi-class classification, and regression
"""
import numpy as np
import copy

from models.layer import Dense, ActivationLayer
from utils.activation import get_activation, softmax
from utils.loss import get_loss
from reg.l2 import l2_penalty, l2_gradient
from reg.dropout import Dropout
from reg.early_stop import EarlyStopping


class MLP:
    """
    Multi-Layer Perceptron built from Dense + Activation layer pairs.

    Args:
        layer_sizes      : list of ints, e.g. [2, 16, 16, 1] means
                           input_dim=2, two hidden layers of 16 neurons,
                           output_dim=1.
        activations      : list of activation names, one per hidden+output layer.
                           e.g. ['relu', 'relu', 'sigmoid']
        loss             : loss function name ('mse', 'binary_cross_entropy',
                           'categorical_cross_entropy')
        learning_rate    : float
        lambda_          : L2 regularization strength (0 = disabled)
        dropout_rate     : fraction of neurons to DROP per hidden layer (0 = disabled)
        batch_size       : int or None. None → full-batch gradient descent.
        optimizer        : 'sgd' (only option for now; mini-batch is handled internally)
    """

    def __init__(self,
                 layer_sizes,
                 activations,
                 loss='binary_cross_entropy',
                 learning_rate=0.01,
                 lambda_=0.0,
                 dropout_rate=0.0,
                 batch_size=None):
        assert len(layer_sizes) >= 2, "Need at least input and output size"
        assert len(activations) == len(layer_sizes) - 1, \
            "One activation per layer transition required"

        self.layer_sizes = layer_sizes
        self.activations = activations
        self.loss_name = loss
        self.learning_rate = learning_rate
        self.lambda_ = lambda_
        self.dropout_rate = dropout_rate
        self.batch_size = batch_size

        self.loss_fn, self.loss_derivative = get_loss(loss)

        # Build layer sequence
        self._build_layers()

        # Training history
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []

    # ──────────────────────────────────────────────────────────
    # Construction
    # ──────────────────────────────────────────────────────────

    def _build_layers(self):
        """Create the ordered list of Dense + Activation + optional Dropout layers."""
        self.layers = []
        self.dense_layers = []     # References for param updates and L2
        self.dropout_layers = []   # References for toggling train/inference

        n_transitions = len(self.layer_sizes) - 1
        for i in range(n_transitions):
            in_size = self.layer_sizes[i]
            out_size = self.layer_sizes[i + 1]
            act_name = self.activations[i]

            act_fn, act_deriv = get_activation(act_name)

            dense = Dense(in_size, out_size)
            activation = ActivationLayer(act_fn, act_deriv)
            self.layers.append(dense)
            self.layers.append(activation)
            self.dense_layers.append(dense)

            # Add dropout after each hidden layer (not the output layer)
            is_output = (i == n_transitions - 1)
            if self.dropout_rate > 0 and not is_output:
                drop = Dropout(self.dropout_rate)
                self.layers.append(drop)
                self.dropout_layers.append(drop)

    # ──────────────────────────────────────────────────────────
    # Forward & Backward
    # ──────────────────────────────────────────────────────────

    def forward(self, X, training=True):
        """Run X through all layers and return output."""
        out = X
        for layer in self.layers:
            if isinstance(layer, Dropout):
                out = layer.forward(out, training=training)
            else:
                out = layer.forward(out, training=training)
        return out

    def backward(self, grad):
        """Backprop gradient through all layers in reverse."""
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def _update_weights(self):
        """Apply SGD update to all Dense layers, with optional L2 gradient."""
        for dense in self.dense_layers:
            if self.lambda_ > 0:
                dense.dweights += l2_gradient(dense.weights, self.lambda_)
            dense.weights -= self.learning_rate * dense.dweights
            dense.bias -= self.learning_rate * dense.dbias

    # ──────────────────────────────────────────────────────────
    # Training
    # ──────────────────────────────────────────────────────────

    def train(self, X_train, y_train,
              epochs=100,
              X_val=None, y_val=None,
              early_stopper=None,
              verbose=True,
              epoch_callback=None):
        """
        Train the network.

        Args:
            epoch_callback: callable(epoch, train_loss, val_loss, train_acc, val_acc)
                            Called at the end of every epoch — use this to update
                            live Streamlit charts.
        Returns:
            History dict with lists of losses and accuracies.
        """
        self.train_losses.clear()
        self.val_losses.clear()
        self.train_accs.clear()
        self.val_accs.clear()

        n = X_train.shape[0]
        use_minibatch = self.batch_size is not None and self.batch_size < n

        for epoch in range(1, epochs + 1):
            # Shuffle training data each epoch
            idx = np.random.permutation(n)
            X_shuf, y_shuf = X_train[idx], y_train[idx]

            if use_minibatch:
                batches = self._get_batches(X_shuf, y_shuf)
            else:
                batches = [(X_shuf, y_shuf)]

            for X_b, y_b in batches:
                # Forward
                output = self.forward(X_b, training=True)
                # Loss gradient
                grad = self.loss_derivative(output, y_b)
                # Backward
                self.backward(grad)
                # Update
                self._update_weights()

            # ── Evaluate on full training set (no dropout) ──
            train_out = self.forward(X_train, training=False)
            t_loss = self.loss_fn(train_out, y_train)
            if self.lambda_ > 0:
                w_list = [d.weights for d in self.dense_layers]
                t_loss += l2_penalty(w_list, self.lambda_)
            t_acc = self._accuracy(train_out, y_train)

            # ── Evaluate on validation set ──
            if X_val is not None and y_val is not None:
                val_out = self.forward(X_val, training=False)
                v_loss = self.loss_fn(val_out, y_val)
                v_acc = self._accuracy(val_out, y_val)
            else:
                v_loss, v_acc = None, None

            self.train_losses.append(t_loss)
            self.val_losses.append(v_loss)
            self.train_accs.append(t_acc)
            self.val_accs.append(v_acc)

            if verbose and epoch % max(1, epochs // 10) == 0:
                msg = f"Epoch {epoch}/{epochs} | Loss: {t_loss:.4f} | Acc: {t_acc:.4f}"
                if v_loss is not None:
                    msg += f" | Val Loss: {v_loss:.4f} | Val Acc: {v_acc:.4f}"
                print(msg)

            if epoch_callback:
                epoch_callback(epoch, t_loss, v_loss, t_acc, v_acc)

            # Early stopping
            if early_stopper is not None:
                monitor = v_loss if v_loss is not None else t_loss
                if early_stopper(monitor, epoch, model=self):
                    if verbose:
                        print(f"Early stopping at epoch {epoch} "
                              f"(best epoch: {early_stopper.best_epoch})")
                    break

        return {
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "train_accs": self.train_accs,
            "val_accs": self.val_accs,
        }

    def predict(self, X):
        """Run forward pass in inference mode."""
        return self.forward(X, training=False)

    def predict_classes(self, X):
        """Return class labels (int array)."""
        out = self.predict(X)
        if out.shape[1] == 1:
            return (out.ravel() >= 0.5).astype(int)
        return np.argmax(out, axis=1)

    # ──────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────

    def _get_batches(self, X, y):
        n = X.shape[0]
        for start in range(0, n, self.batch_size):
            end = start + self.batch_size
            yield X[start:end], y[start:end]

    def _accuracy(self, output, y_true):
        """Classification accuracy (skipped for regression / MSE loss)."""
        if self.loss_name == 'mse':
            return None  # Accuracy meaningless for regression
        if output.shape[1] == 1:
            pred = (output.ravel() >= 0.5).astype(int)
            true = y_true.ravel().astype(int)
        else:
            pred = np.argmax(output, axis=1)
            true = y_true.ravel().astype(int) if y_true.ndim == 1 \
                else np.argmax(y_true, axis=1)
        return float(np.mean(pred == true))

    # ──────────────────────────────────────────────────────────
    # Weight persistence (for early stopping restore)
    # ──────────────────────────────────────────────────────────

    def get_weights(self):
        """Return a deep-copy snapshot of all Dense layer weights."""
        return [(copy.deepcopy(d.weights), copy.deepcopy(d.bias))
                for d in self.dense_layers]

    def set_weights(self, weights):
        """Restore Dense layer weights from a snapshot."""
        for dense, (w, b) in zip(self.dense_layers, weights):
            dense.weights = w
            dense.bias = b
