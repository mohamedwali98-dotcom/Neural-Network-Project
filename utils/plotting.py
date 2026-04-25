"""
Plotting utilities for the Neural Network Educational Platform.

Functions:
  - plot_decision_boundary : 2D colored background decision region
  - plot_training_curves   : Loss and accuracy vs epochs using Plotly
  - plot_confusion_matrix  : Annotated heatmap
"""
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ────────────────────────────────────────────────────────────
# Decision Boundary
# ────────────────────────────────────────────────────────────

def plot_decision_boundary(model, X, y, resolution=300, title="Decision Boundary"):
    """
    Generate a 2-D decision boundary plot.

    Args:
        model      : trained model with a .predict(X) method
        X          : numpy array of shape (n, 2)
        y          : numpy label array of shape (n,)
        resolution : grid resolution (higher = smoother, slower)
        title      : plot title string

    Returns:
        Plotly Figure
    """
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, resolution),
        np.linspace(y_min, y_max, resolution),
    )
    grid = np.c_[xx.ravel(), yy.ravel()]

    # Model output
    Z = model.predict(grid)
    if Z.ndim == 2 and Z.shape[1] > 1:
        Z = np.argmax(Z, axis=1)
    else:
        Z = (Z.ravel() >= 0.5).astype(float)
    Z = Z.reshape(xx.shape)

    fig = go.Figure()

    # Background heatmap (decision regions)
    fig.add_trace(go.Heatmap(
        x=np.linspace(x_min, x_max, resolution),
        y=np.linspace(y_min, y_max, resolution),
        z=Z,
        colorscale='RdBu',
        showscale=False,
        opacity=0.6,
    ))

    # Scatter data points
    unique_labels = np.unique(y.ravel().astype(int))
    colors = ['#ef4444', '#3b82f6', '#22c55e', '#f59e0b',
              '#a855f7', '#ec4899', '#14b8a6']
    marker_symbols = ['circle', 'square', 'diamond', 'cross',
                      'x', 'triangle-up', 'triangle-down']

    for cls in unique_labels:
        mask = y.ravel().astype(int) == cls
        fig.add_trace(go.Scatter(
            x=X[mask, 0],
            y=X[mask, 1],
            mode='markers',
            name=f'Class {cls}',
            marker=dict(
                color=colors[cls % len(colors)],
                size=8,
                symbol=marker_symbols[cls % len(marker_symbols)],
                line=dict(color='white', width=1),
            ),
        ))

    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=18)),
        xaxis_title="Feature 1",
        yaxis_title="Feature 2",
        template='plotly_dark',
        paper_bgcolor='rgba(15,15,25,1)',
        plot_bgcolor='rgba(15,15,25,1)',
        legend=dict(
            bgcolor='rgba(30,30,50,0.8)',
            bordercolor='rgba(255,255,255,0.2)',
        ),
        height=480,
        margin=dict(l=40, r=20, t=60, b=40),
    )
    return fig


# ────────────────────────────────────────────────────────────
# Training Curves
# ────────────────────────────────────────────────────────────

def plot_training_curves(history, title="Training Curves"):
    """
    Plot loss and accuracy (if available) vs epochs.

    Args:
        history : dict with keys train_losses, val_losses,
                  train_accs, val_accs  (val_* may contain None)
        title   : plot title

    Returns:
        Plotly Figure
    """
    train_losses = history.get("train_losses", [])
    val_losses   = history.get("val_losses", [])
    train_accs   = history.get("train_accs", [])
    val_accs     = history.get("val_accs", [])

    epochs = list(range(1, len(train_losses) + 1))

    has_acc = any(a is not None for a in train_accs)
    has_val = any(v is not None for v in val_losses)

    rows = 2 if has_acc else 1
    subplot_titles = ["Loss vs Epochs", "Accuracy vs Epochs"] if has_acc else ["Loss vs Epochs"]

    fig = make_subplots(
        rows=rows, cols=1,
        subplot_titles=subplot_titles,
        vertical_spacing=0.12,
    )

    # ── Loss ──
    fig.add_trace(go.Scatter(
        x=epochs, y=train_losses,
        name='Train Loss',
        line=dict(color='#f59e0b', width=2),
        mode='lines',
    ), row=1, col=1)

    if has_val:
        val_clean = [v if v is not None else float('nan') for v in val_losses]
        fig.add_trace(go.Scatter(
            x=epochs, y=val_clean,
            name='Val Loss',
            line=dict(color='#ef4444', width=2, dash='dash'),
            mode='lines',
        ), row=1, col=1)

    # ── Accuracy ──
    if has_acc:
        train_acc_clean = [a if a is not None else float('nan') for a in train_accs]
        fig.add_trace(go.Scatter(
            x=epochs, y=train_acc_clean,
            name='Train Accuracy',
            line=dict(color='#3b82f6', width=2),
            mode='lines',
        ), row=2, col=1)

        if has_val:
            val_acc_clean = [a if a is not None else float('nan') for a in val_accs]
            fig.add_trace(go.Scatter(
                x=epochs, y=val_acc_clean,
                name='Val Accuracy',
                line=dict(color='#a855f7', width=2, dash='dash'),
                mode='lines',
            ), row=2, col=1)

    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=18)),
        template='plotly_dark',
        paper_bgcolor='rgba(15,15,25,1)',
        plot_bgcolor='rgba(15,15,25,1)',
        legend=dict(bgcolor='rgba(30,30,50,0.8)', bordercolor='rgba(255,255,255,0.2)'),
        height=420 * rows,
        margin=dict(l=50, r=20, t=70, b=40),
    )
    fig.update_xaxes(title_text="Epoch", gridcolor='rgba(255,255,255,0.08)')
    fig.update_yaxes(gridcolor='rgba(255,255,255,0.08)')
    return fig


# ────────────────────────────────────────────────────────────
# Confusion Matrix
# ────────────────────────────────────────────────────────────

def plot_confusion_matrix(cm, class_names=None, title="Confusion Matrix"):
    """
    Plot an annotated confusion matrix heatmap.

    Args:
        cm          : 2D numpy array (num_classes × num_classes)
        class_names : list of strings or None
        title       : plot title

    Returns:
        Plotly Figure
    """
    n = cm.shape[0]
    if class_names is None:
        class_names = [str(i) for i in range(n)]

    text = [[str(cm[i, j]) for j in range(n)] for i in range(n)]

    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=class_names,
        y=class_names,
        colorscale='Blues',
        text=text,
        texttemplate="%{text}",
        textfont=dict(size=14),
        showscale=True,
    ))

    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=18)),
        xaxis_title="Predicted",
        yaxis_title="True",
        template='plotly_dark',
        paper_bgcolor='rgba(15,15,25,1)',
        plot_bgcolor='rgba(15,15,25,1)',
        height=400,
        margin=dict(l=60, r=20, t=60, b=60),
    )
    return fig
