"""
Evaluation metrics for classification and regression.
"""
import numpy as np


# ──────────────────────────────────────────────────────────────
# Classification metrics
# ──────────────────────────────────────────────────────────────

def accuracy(y_pred, y_true):
    """Fraction of correctly classified samples."""
    pred_labels = _to_labels(y_pred)
    true_labels = _to_labels(y_true)
    return float(np.mean(pred_labels == true_labels))


def confusion_matrix(y_pred, y_true, num_classes=None):
    """
    Returns a confusion matrix of shape (num_classes, num_classes).
    Entry [i, j] = number of samples with true class i predicted as class j.
    """
    pred_labels = _to_labels(y_pred)
    true_labels = _to_labels(y_true)
    if num_classes is None:
        num_classes = max(int(true_labels.max()), int(pred_labels.max())) + 1
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(true_labels.astype(int), pred_labels.astype(int)):
        cm[t][p] += 1
    return cm


def precision_recall_f1(y_pred, y_true, num_classes=None):
    """
    Per-class and macro-average Precision, Recall, F1.
    Returns dict with keys: precision, recall, f1, macro_precision, macro_recall, macro_f1.
    """
    cm = confusion_matrix(y_pred, y_true, num_classes)
    n = cm.shape[0]
    precision = np.zeros(n)
    recall = np.zeros(n)
    f1 = np.zeros(n)
    for i in range(n):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        precision[i] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall[i] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1[i] = (2 * precision[i] * recall[i] / (precision[i] + recall[i])
                 if (precision[i] + recall[i]) > 0 else 0.0)
    return {
        "precision": precision.tolist(),
        "recall": recall.tolist(),
        "f1": f1.tolist(),
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "macro_f1": float(f1.mean()),
    }


# ──────────────────────────────────────────────────────────────
# Regression metrics
# ──────────────────────────────────────────────────────────────

def mse_metric(y_pred, y_true):
    return float(np.mean((y_pred - y_true) ** 2))


def rmse_metric(y_pred, y_true):
    return float(np.sqrt(mse_metric(y_pred, y_true)))


def r2_score(y_pred, y_true):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────

def _to_labels(y):
    """Convert probability outputs to class labels."""
    y = np.array(y)
    if y.ndim == 1 or y.shape[1] == 1:
        # Binary: threshold at 0.5
        return (y.ravel() >= 0.5).astype(int)
    else:
        # Multi-class: argmax
        return np.argmax(y, axis=1)
