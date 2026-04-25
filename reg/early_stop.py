"""
Early stopping tracker.
"""


class EarlyStopping:
    """
    Stops training when monitored metric stops improving.

    Args:
        patience: number of epochs to wait after last improvement.
        min_delta: minimum change to qualify as improvement.
        mode: 'min' (loss) or 'max' (accuracy).
        restore_best: if True, revert to best weights when stopping.
    """

    def __init__(self, patience=10, min_delta=1e-4, mode="min", restore_best=True):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best = restore_best
        self.best_value = float("inf") if mode == "min" else float("-inf")
        self.counter = 0
        self.best_weights = None
        self.stopped = False
        self.best_epoch = 0

    def __call__(self, value, epoch, model=None):
        """
        Call each epoch.
        Returns True if training should stop.
        """
        improved = (
            (value < self.best_value - self.min_delta) if self.mode == "min"
            else (value > self.best_value + self.min_delta)
        )
        if improved:
            self.best_value = value
            self.counter = 0
            self.best_epoch = epoch
            if self.restore_best and model is not None:
                import copy
                self.best_weights = copy.deepcopy(model.get_weights())
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stopped = True
                if self.restore_best and model is not None and self.best_weights is not None:
                    model.set_weights(self.best_weights)
                return True
        return False

    def reset(self):
        self.best_value = float("inf") if self.mode == "min" else float("-inf")
        self.counter = 0
        self.best_weights = None
        self.stopped = False
        self.best_epoch = 0
