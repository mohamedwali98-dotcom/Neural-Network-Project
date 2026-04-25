"""
Perceptron models.

  - HistoricalPerceptron : Classic Rosenblatt perceptron with step activation
                           and a simple online delta-rule weight update.
  - ModernPerceptron     : Single-layer, configurable activation, SGD via
                           gradient descent (no hidden layers).
"""
import numpy as np
from utils.activation import get_activation, step


class HistoricalPerceptron:
    """
    Rosenblatt's original (1957) perceptron.

    Uses the step activation function and the perceptron learning rule:
        w ← w + lr * (y_true - y_pred) * x
    Works ONLY for linearly separable data.
    """

    def __init__(self, input_size, learning_rate=0.01):
        self.lr = learning_rate
        self.weights = np.zeros(input_size)
        self.bias = 0.0
        self.train_losses = []   # fraction of misclassified samples per epoch
        self.train_accs = []

    def _predict_raw(self, X):
        return np.dot(X, self.weights) + self.bias

    def predict(self, X):
        return step(self._predict_raw(X))

    def train(self, X_train, y_train, epochs=100,
              X_val=None, y_val=None,
              verbose=True, epoch_callback=None):
        """Online perceptron learning rule."""
        self.train_losses.clear()
        self.train_accs.clear()
        n = X_train.shape[0]
        y = y_train.ravel()

        for epoch in range(1, epochs + 1):
            errors = 0
            for xi, yi in zip(X_train, y):
                pred = self.predict(xi.reshape(1, -1)).item()
                delta = yi - pred
                if delta != 0:
                    self.weights += self.lr * delta * xi
                    self.bias += self.lr * delta
                    errors += 1

            acc = 1.0 - errors / n
            loss = errors / n   # 0-1 loss
            self.train_losses.append(loss)
            self.train_accs.append(acc)

            if verbose and epoch % max(1, epochs // 10) == 0:
                print(f"Epoch {epoch}/{epochs} | Misclassified: {errors}/{n} | Acc: {acc:.4f}")

            if epoch_callback:
                epoch_callback(epoch, loss, None, acc, None)

        return {
            "train_losses": self.train_losses,
            "val_losses": [None] * len(self.train_losses),
            "train_accs": self.train_accs,
            "val_accs": [None] * len(self.train_accs),
        }

    def get_weights(self):
        return (self.weights.copy(), self.bias)

    def set_weights(self, weights):
        self.weights, self.bias = weights[0].copy(), weights[1]


class ModernPerceptron:
    """
    A single-layer perceptron trained with gradient descent.
    Uses any differentiable activation function and a proper loss function.
    No hidden layers — demonstrates what a perceptron can and cannot learn.
    """

    def __init__(self, input_size, output_size=1,
                 activation='sigmoid',
                 loss='binary_cross_entropy',
                 learning_rate=0.01):
        from utils.loss import get_loss
        self.lr = learning_rate
        self.act_name = activation
        self.loss_name = loss
        act_fn, act_deriv = get_activation(activation)
        self.act_fn = act_fn
        self.act_deriv = act_deriv
        self.loss_fn, self.loss_deriv = get_loss(loss)

        # Parameters
        self.weights = np.random.randn(input_size, output_size) * 0.01
        self.bias = np.zeros((1, output_size))

        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []

    def forward(self, X):
        z = np.dot(X, self.weights) + self.bias
        return self.act_fn(z)

    def predict(self, X):
        return self.forward(X)

    def predict_classes(self, X):
        out = self.predict(X)
        if out.shape[1] == 1:
            return (out.ravel() >= 0.5).astype(int)
        return np.argmax(out, axis=1)

    def train(self, X_train, y_train, epochs=100,
              X_val=None, y_val=None,
              verbose=True, epoch_callback=None):
        """Gradient descent training loop."""
        self.train_losses.clear()
        self.val_losses.clear()
        self.train_accs.clear()
        self.val_accs.clear()

        n = X_train.shape[0]
        y = y_train.reshape(n, -1)

        for epoch in range(1, epochs + 1):
            # Forward
            z = np.dot(X_train, self.weights) + self.bias
            output = self.act_fn(z)

            # Loss + gradient
            loss = self.loss_fn(output, y)
            dL_dout = self.loss_deriv(output, y)

            # Backprop through activation
            dout_dz = self.act_deriv(z)
            dz = dL_dout * dout_dz

            # Parameter gradients
            dW = np.dot(X_train.T, dz)
            db = np.sum(dz, axis=0, keepdims=True)

            # Update
            self.weights -= self.lr * dW
            self.bias -= self.lr * db

            # Metrics
            t_acc = self._accuracy(output, y)
            self.train_losses.append(float(loss))
            self.train_accs.append(t_acc)

            if X_val is not None and y_val is not None:
                val_out = self.forward(X_val)
                v_loss = float(self.loss_fn(val_out, y_val.reshape(X_val.shape[0], -1)))
                v_acc = self._accuracy(val_out, y_val.reshape(X_val.shape[0], -1))
            else:
                v_loss, v_acc = None, None
            self.val_losses.append(v_loss)
            self.val_accs.append(v_acc)

            if verbose and epoch % max(1, epochs // 10) == 0:
                print(f"Epoch {epoch}/{epochs} | Loss: {loss:.4f} | Acc: {t_acc:.4f}")

            if epoch_callback:
                epoch_callback(epoch, float(loss), v_loss, t_acc, v_acc)

        return {
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "train_accs": self.train_accs,
            "val_accs": self.val_accs,
        }

    def _accuracy(self, output, y_true):
        if self.loss_name == 'mse':
            return None
        if output.shape[1] == 1:
            pred = (output.ravel() >= 0.5).astype(int)
            true = y_true.ravel().astype(int)
        else:
            pred = np.argmax(output, axis=1)
            true = y_true.ravel().astype(int)
        return float(np.mean(pred == true))

    def get_weights(self):
        import copy
        return (copy.deepcopy(self.weights), copy.deepcopy(self.bias))

    def set_weights(self, weights):
        import copy
        self.weights = copy.deepcopy(weights[0])
        self.bias = copy.deepcopy(weights[1])
