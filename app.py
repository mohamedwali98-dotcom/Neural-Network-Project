"""
Neural Network Educational Platform
Streamlit Application
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from sklearn.datasets import make_moons, make_circles, make_blobs
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

from models.mlp import MLP
from models.perceptron import HistoricalPerceptron, ModernPerceptron
from utils.metrics import accuracy, confusion_matrix, precision_recall_f1, mse_metric, rmse_metric, r2_score
from utils.plotting import plot_decision_boundary, plot_training_curves, plot_confusion_matrix
from reg.early_stop import EarlyStopping

# ─────────────────────────────────────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NeuralVis — Neural Network Platform",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CSS Styling
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: linear-gradient(135deg, #0a0a14 0%, #0f0f1e 50%, #090912 100%);
    color: #e2e8f0;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d0d1f 0%, #0a0a18 100%);
    border-right: 1px solid rgba(99,102,241,0.25);
}
section[data-testid="stSidebar"] .block-container {
    padding-top: 1.5rem;
}

/* Header gradient */
.hero-header {
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #a855f7 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 2.8rem;
    font-weight: 700;
    text-align: center;
    margin-bottom: 0.2rem;
    line-height: 1.15;
}
.hero-sub {
    text-align: center;
    color: #94a3b8;
    font-size: 1rem;
    margin-bottom: 2rem;
}

/* Metric cards */
.metric-card {
    background: linear-gradient(135deg, rgba(99,102,241,0.12) 0%, rgba(139,92,246,0.08) 100%);
    border: 1px solid rgba(99,102,241,0.3);
    border-radius: 12px;
    padding: 1rem 1.2rem;
    text-align: center;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.metric-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 30px rgba(99,102,241,0.25);
}
.metric-label {
    font-size: 0.78rem;
    color: #7c3aed;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 0.3rem;
}
.metric-value {
    font-size: 1.8rem;
    font-weight: 700;
    color: #e2e8f0;
    font-family: 'JetBrains Mono', monospace;
}

/* Section headers */
.section-header {
    font-size: 1.1rem;
    font-weight: 600;
    color: #a78bfa;
    border-left: 3px solid #7c3aed;
    padding-left: 0.6rem;
    margin: 1.2rem 0 0.8rem;
}

/* Status badge */
.badge {
    display: inline-block;
    padding: 0.2rem 0.6rem;
    border-radius: 99px;
    font-size: 0.75rem;
    font-weight: 600;
}
.badge-ready { background: rgba(34,197,94,0.15); color: #22c55e; border: 1px solid rgba(34,197,94,0.3); }
.badge-training { background: rgba(245,158,11,0.15); color: #f59e0b; border: 1px solid rgba(245,158,11,0.3); }
.badge-idle { background: rgba(148,163,184,0.12); color: #94a3b8; border: 1px solid rgba(148,163,184,0.2); }

/* Architecture viz */
.layer-pill {
    background: linear-gradient(90deg, rgba(99,102,241,0.2), rgba(139,92,246,0.15));
    border: 1px solid rgba(99,102,241,0.35);
    border-radius: 8px;
    padding: 0.5rem 1rem;
    font-size: 0.85rem;
    font-family: 'JetBrains Mono', monospace;
    color: #c4b5fd;
    margin: 0.2rem 0;
}

/* Divider */
.fancy-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(99,102,241,0.5), transparent);
    margin: 1.5rem 0;
}

/* Train button */
div[data-testid="stButton"] > button {
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 0.6rem 1.5rem;
    font-weight: 600;
    font-size: 1rem;
    transition: all 0.25s ease;
    width: 100%;
    box-shadow: 0 4px 15px rgba(99,102,241,0.35);
}
div[data-testid="stButton"] > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(99,102,241,0.5);
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Session State
# ─────────────────────────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = None
if "model" not in st.session_state:
    st.session_state.model = None
if "X_train" not in st.session_state:
    st.session_state.X_train = None
if "X_test" not in st.session_state:
    st.session_state.X_test = None
if "y_train" not in st.session_state:
    st.session_state.y_train = None
if "y_test" not in st.session_state:
    st.session_state.y_test = None
if "task_type" not in st.session_state:
    st.session_state.task_type = "binary"
if "feature_names" not in st.session_state:
    st.session_state.feature_names = None
if "scaler" not in st.session_state:
    st.session_state.scaler = None
if "y_scaler" not in st.session_state:
    st.session_state.y_scaler = None

# ─────────────────────────────────────────────────────────────────────────────
# Hero Header
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<h1 class="hero-header">🧠 NeuralVis</h1>', unsafe_allow_html=True)
st.markdown('<p class="hero-sub">An Interactive Neural Network Educational Platform</p>', unsafe_allow_html=True)
st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📊 Data Configuration")

    data_source = st.selectbox(
        "Data Source",
        ["📁 Upload CSV", "🌙 Make Moons", "⭕ Make Circles", "🌐 Make Blobs (Multi-class)"],
        key="data_source_selector",
    )

    test_size = st.slider("Test Split (%)", 10, 40, 20, 5) / 100.0
    random_seed = st.number_input("Random Seed", value=42, min_value=0, max_value=9999, step=1)

    # Dataset-specific options
    if data_source in ["🌙 Make Moons", "⭕ Make Circles"]:
        n_samples = st.slider("Samples", 100, 1000, 300, 50)
        noise = st.slider("Noise", 0.0, 0.5, 0.15, 0.05)
    elif data_source == "🌐 Make Blobs (Multi-class)":
        n_samples = st.slider("Samples", 100, 1000, 300, 50)
        n_centers = st.slider("Number of Classes", 2, 6, 3)

    st.markdown("---")
    st.markdown("## 🤖 Model Configuration")

    model_type = st.selectbox(
        "Model",
        ["MLP (Multi-Layer Perceptron)", "Modern Perceptron", "Historical Perceptron"],
        key="model_type_selector",
    )

    st.markdown("---")
    st.markdown("## ⚙️ Hyperparameters")

    learning_rate = st.select_slider(
        "Learning Rate",
        options=[0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5],
        value=0.01,
    )
    n_epochs = st.slider("Epochs", 10, 500, 100, 10)

    if model_type == "MLP (Multi-Layer Perceptron)":
        st.markdown("### 🏗️ Architecture")
        n_hidden = st.slider("Hidden Layers", 1, 5, 2)
        hidden_neurons = []
        for i in range(n_hidden):
            neurons = st.slider(f"Neurons in Layer {i+1}", 2, 256, 16, 2)
            hidden_neurons.append(neurons)

        hidden_activation = st.selectbox("Hidden Activation", ["relu", "tanh", "sigmoid"])
        use_minibatch = st.checkbox("Mini-batch Gradient Descent", value=False)
        batch_size = None
        if use_minibatch:
            batch_size = st.slider("Batch Size", 8, 256, 32, 8)

        st.markdown("### 🛡️ Regularization")
        lambda_ = st.select_slider(
            "L2 Lambda",
            options=[0.0, 0.0001, 0.001, 0.01, 0.1, 1.0],
            value=0.0,
        )
        dropout_rate = st.select_slider(
            "Dropout Rate",
            options=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
            value=0.0,
        )
        use_early_stop = st.checkbox("Early Stopping", value=False)
        patience = 15
        if use_early_stop:
            patience = st.slider("Patience (epochs)", 5, 50, 15)

    st.markdown("---")
    train_btn = st.button("🚀 Train Model", use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# Data Loading Logic
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data
def load_synthetic(source, n_samples, noise=0.15, seed=42, n_centers=3):
    np.random.seed(seed)
    if source == "🌙 Make Moons":
        X, y = make_moons(n_samples=n_samples, noise=noise, random_state=seed)
        task = "binary"
    elif source == "⭕ Make Circles":
        X, y = make_circles(n_samples=n_samples, noise=noise, factor=0.5, random_state=seed)
        task = "binary"
    else:
        X, y = make_blobs(n_samples=n_samples, centers=n_centers, random_state=seed)
        task = "multiclass"
    return X, y, task


# ─────────────────────────────────────────────────────────────────────────────
# Load and split data
# ─────────────────────────────────────────────────────────────────────────────
if data_source in ["🌙 Make Moons", "⭕ Make Circles"]:
    X, y, task_type, feat_names = load_synthetic(data_source, n_samples, noise, random_seed), None, None, None
    X, y, task_type = load_synthetic(data_source, n_samples, noise, random_seed)
    feat_names = ["Feature 1", "Feature 2"]
elif data_source == "🌐 Make Blobs (Multi-class)":
    X, y, task_type = load_synthetic(data_source, n_samples, seed=random_seed, n_centers=n_centers)
    feat_names = ["Feature 1", "Feature 2"]
else:
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded is None:
        st.info("👈 Upload a CSV file or select a synthetic dataset from the sidebar to get started.")
        st.stop()
    df = pd.read_csv(uploaded)
    target_col = st.selectbox("Select Target Column", df.columns.tolist())
    feature_cols = [c for c in df.columns if c != target_col]

    # Auto-encode any non-numeric feature columns
    df_features = df[feature_cols].copy()
    for col in df_features.columns:
        if not pd.api.types.is_numeric_dtype(df_features[col]):
            _enc = LabelEncoder()
            df_features[col] = _enc.fit_transform(df_features[col].astype(str))
    X = df_features.values.astype(float)

    task_type_ui = st.selectbox("Task Type", ["Binary Classification", "Multi-class Classification", "Regression"])
    if task_type_ui == "Binary Classification":
        task_type = "binary"
    elif task_type_ui == "Multi-class Classification":
        task_type = "multiclass"
    else:
        task_type = "regression"

    # Encode target based on task type
    y_raw = df[target_col].values
    if task_type in ["binary", "multiclass"]:
        le = LabelEncoder()
        y = le.fit_transform(y_raw).astype(float)
    else:
        try:
            y = y_raw.astype(float)
        except ValueError:
            st.error("Target column must be numeric for Regression!")
            st.stop()

    feat_names = feature_cols

# Normalize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

st.session_state.scaler = scaler
st.session_state.feature_names = feat_names

y_scaler = None
if task_type == "regression":
    y_scaler = StandardScaler()
    y = y_scaler.fit_transform(y.reshape(-1, 1)).flatten()

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=test_size, random_state=int(random_seed)
)
st.session_state.X_train = X_train
st.session_state.X_test = X_test
st.session_state.y_train = y_train
st.session_state.y_test = y_test
st.session_state.task_type = task_type
st.session_state.y_scaler = y_scaler

# ─────────────────────────────────────────────────────────────────────────────
# Data Preview Section
# ─────────────────────────────────────────────────────────────────────────────
n_classes = int(y.max()) + 1

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Total Samples</div>
        <div class="metric-value">{len(X)}</div>
    </div>""", unsafe_allow_html=True)
with col2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Training Samples</div>
        <div class="metric-value">{len(X_train)}</div>
    </div>""", unsafe_allow_html=True)
with col3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Test Samples</div>
        <div class="metric-value">{len(X_test)}</div>
    </div>""", unsafe_allow_html=True)
with col4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Classes</div>
        <div class="metric-value">{n_classes}</div>
    </div>""", unsafe_allow_html=True)

# ── Dataset scatter preview ──
if X_scaled.shape[1] >= 2:
    st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-header">📍 Dataset Preview</div>', unsafe_allow_html=True)
    colors_list = ['#ef4444','#3b82f6','#22c55e','#f59e0b','#a855f7','#ec4899']
    fig_data = go.Figure()
    for cls in range(n_classes):
        mask = y.astype(int) == cls
        fig_data.add_trace(go.Scatter(
            x=X_scaled[mask, 0], y=X_scaled[mask, 1],
            mode='markers',
            name=f'Class {cls}',
            marker=dict(color=colors_list[cls % len(colors_list)], size=7,
                        line=dict(color='white', width=0.5)),
        ))
    fig_data.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(15,15,25,1)',
        plot_bgcolor='rgba(15,15,25,1)',
        height=340,
        margin=dict(l=40, r=20, t=20, b=40),
        xaxis=dict(title=feat_names[0] if feat_names else "Feature 1", gridcolor='rgba(255,255,255,0.06)'),
        yaxis=dict(title=feat_names[1] if feat_names else "Feature 2", gridcolor='rgba(255,255,255,0.06)'),
        legend=dict(bgcolor='rgba(30,30,50,0.8)'),
    )
    st.plotly_chart(fig_data, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# Architecture Diagram
# ─────────────────────────────────────────────────────────────────────────────
if model_type == "MLP (Multi-Layer Perceptron)":
    st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-header">🏗️ Network Architecture</div>', unsafe_allow_html=True)

    input_dim = X_train.shape[1]
    output_dim = n_classes if task_type == "multiclass" else 1
    output_act = "softmax" if task_type == "multiclass" else "sigmoid"

    arch_cols = st.columns(2 + n_hidden)
    with arch_cols[0]:
        st.markdown(f'<div class="layer-pill">📥 Input<br/>{input_dim} features</div>', unsafe_allow_html=True)
    for i, neurons in enumerate(hidden_neurons):
        with arch_cols[i + 1]:
            st.markdown(f'<div class="layer-pill">⚡ Hidden {i+1}<br/>{neurons} × {hidden_activation}</div>', unsafe_allow_html=True)
    with arch_cols[-1]:
        st.markdown(f'<div class="layer-pill">📤 Output<br/>{output_dim} × {output_act}</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Training Logic
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)

if train_btn:
    input_dim = X_train.shape[1]
    n_cls = int(y.max()) + 1

    # Determine loss and output settings
    if task_type == "multiclass":
        loss_name = "categorical_cross_entropy"
        output_size = n_cls
        output_activation = "softmax"
        y_train_m = y_train.astype(int)
        y_test_m = y_test.astype(int)
    elif task_type == "binary":
        loss_name = "binary_cross_entropy"
        output_size = 1
        output_activation = "sigmoid"
        y_train_m = y_train.reshape(-1, 1)
        y_test_m = y_test.reshape(-1, 1)
    else:
        loss_name = "mse"
        output_size = 1
        output_activation = "linear"
        y_train_m = y_train.reshape(-1, 1)
        y_test_m = y_test.reshape(-1, 1)

    # ── Build model ──
    st.markdown('<div class="section-header">🔥 Training Progress</div>', unsafe_allow_html=True)

    live_col1, live_col2 = st.columns(2)
    loss_chart = live_col1.empty()
    acc_chart  = live_col2.empty()
    status_bar = st.empty()
    prog_bar   = st.progress(0)

    live_train_losses, live_val_losses = [], []
    live_train_accs,   live_val_accs   = [], []

    def epoch_callback(epoch, t_loss, v_loss, t_acc, v_acc):
        live_train_losses.append(t_loss)
        live_val_losses.append(v_loss)
        live_train_accs.append(t_acc)
        live_val_accs.append(v_acc)
        prog_bar.progress(epoch / n_epochs)
        status_msg = f"Epoch {epoch}/{n_epochs} | Train Loss: {t_loss:.4f}"
        if t_acc is not None:
            status_msg += f" | Train Acc: {t_acc:.4f}"
        if v_loss is not None:
            status_msg += f" | Val Loss: {v_loss:.4f}"
        status_bar.markdown(f'<span class="badge badge-training">⏳ {status_msg}</span>', unsafe_allow_html=True)

        if epoch % max(1, n_epochs // 30) == 0:
            fig_l = go.Figure()
            epochs_so_far = list(range(1, len(live_train_losses)+1))
            fig_l.add_trace(go.Scatter(x=epochs_so_far, y=live_train_losses,
                                        name='Train', line=dict(color='#f59e0b')))
            if any(v is not None for v in live_val_losses):
                clean_v = [v if v is not None else float('nan') for v in live_val_losses]
                fig_l.add_trace(go.Scatter(x=epochs_so_far, y=clean_v,
                                            name='Val', line=dict(color='#ef4444', dash='dash')))
            fig_l.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)',
                                  plot_bgcolor='rgba(0,0,0,0)', height=250,
                                  margin=dict(l=30, r=10, t=30, b=30),
                                  title="Loss", legend=dict(orientation='h'))
            loss_chart.plotly_chart(fig_l, use_container_width=True)

            if any(a is not None for a in live_train_accs):
                fig_a = go.Figure()
                clean_a = [a if a is not None else float('nan') for a in live_train_accs]
                fig_a.add_trace(go.Scatter(x=epochs_so_far, y=clean_a,
                                            name='Train Acc', line=dict(color='#3b82f6')))
                if any(a is not None for a in live_val_accs):
                    clean_va = [a if a is not None else float('nan') for a in live_val_accs]
                    fig_a.add_trace(go.Scatter(x=epochs_so_far, y=clean_va,
                                                name='Val Acc', line=dict(color='#a855f7', dash='dash')))
                fig_a.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)',
                                     plot_bgcolor='rgba(0,0,0,0)', height=250,
                                     margin=dict(l=30, r=10, t=30, b=30),
                                     title="Accuracy", legend=dict(orientation='h'))
                acc_chart.plotly_chart(fig_a, use_container_width=True)

    if model_type == "Historical Perceptron":
        model = HistoricalPerceptron(input_dim, learning_rate=learning_rate)
        history = model.train(X_train, y_train, epochs=n_epochs,
                              X_val=X_test, y_val=y_test,
                              verbose=False, epoch_callback=epoch_callback)

    elif model_type == "Modern Perceptron":
        model = ModernPerceptron(input_dim, output_size=output_size,
                                  activation=output_activation,
                                  loss=loss_name,
                                  learning_rate=learning_rate)
        history = model.train(X_train, y_train_m, epochs=n_epochs,
                              X_val=X_test, y_val=y_test_m,
                              verbose=False, epoch_callback=epoch_callback)

    else:  # MLP
        layer_sizes = [input_dim] + hidden_neurons + [output_size]
        activations = [hidden_activation] * n_hidden + [output_activation]

        model = MLP(
            layer_sizes=layer_sizes,
            activations=activations,
            loss=loss_name,
            learning_rate=learning_rate,
            lambda_=lambda_,
            dropout_rate=dropout_rate,
            batch_size=batch_size,
        )
        early_stopper = EarlyStopping(patience=patience) if use_early_stop else None
        history = model.train(X_train, y_train_m, epochs=n_epochs,
                              X_val=X_test, y_val=y_test_m,
                              early_stopper=early_stopper,
                              verbose=False,
                              epoch_callback=epoch_callback)

    prog_bar.progress(1.0)
    status_bar.markdown('<span class="badge badge-ready">✅ Training Complete</span>', unsafe_allow_html=True)

    st.session_state.model = model
    st.session_state.history = history

# ─────────────────────────────────────────────────────────────────────────────
# Results Section (post-training)
# ─────────────────────────────────────────────────────────────────────────────
if st.session_state.model is not None and st.session_state.history is not None:
    model   = st.session_state.model
    history = st.session_state.history
    X_tr    = st.session_state.X_train
    X_te    = st.session_state.X_test
    y_tr    = st.session_state.y_train
    y_te    = st.session_state.y_test
    task    = st.session_state.task_type

    st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-header">📊 Training Results</div>', unsafe_allow_html=True)

    # ── Final training curves ──
    fig_curves = plot_training_curves(history, title="Full Training History")
    st.plotly_chart(fig_curves, use_container_width=True)

    # ── Evaluation metrics ──
    st.markdown('<div class="section-header">🎯 Evaluation Metrics</div>', unsafe_allow_html=True)

    def get_preds(m, X, task, n_cls):
        out = m.predict(X)
        if task == "multiclass":
            return np.argmax(out, axis=1) if out.ndim == 2 and out.shape[1] > 1 else out.ravel().astype(int)
        elif task == "binary":
            return (out.ravel() >= 0.5).astype(int)
        else:
            return out.ravel()

    n_cls_eval = int(y.max()) + 1
    train_preds = get_preds(model, X_tr, task, n_cls_eval)
    test_preds  = get_preds(model, X_te, task, n_cls_eval)

    if task in ["binary", "multiclass"]:
        train_acc = float(np.mean(train_preds == y_tr.astype(int)))
        test_acc  = float(np.mean(test_preds  == y_te.astype(int)))

        mc1, mc2, mc3, mc4 = st.columns(4)
        with mc1:
            st.markdown(f'<div class="metric-card"><div class="metric-label">Train Accuracy</div><div class="metric-value">{train_acc:.1%}</div></div>', unsafe_allow_html=True)
        with mc2:
            st.markdown(f'<div class="metric-card"><div class="metric-label">Test Accuracy</div><div class="metric-value">{test_acc:.1%}</div></div>', unsafe_allow_html=True)

        # Precision / Recall / F1
        prf = precision_recall_f1(
            test_preds.reshape(-1, 1) if task == "binary" else test_preds,
            y_te.reshape(-1, 1) if task == "binary" else y_te,
            num_classes=n_cls_eval
        )
        with mc3:
            st.markdown(f'<div class="metric-card"><div class="metric-label">Macro F1</div><div class="metric-value">{prf["macro_f1"]:.3f}</div></div>', unsafe_allow_html=True)
        with mc4:
            st.markdown(f'<div class="metric-card"><div class="metric-label">Macro Precision</div><div class="metric-value">{prf["macro_precision"]:.3f}</div></div>', unsafe_allow_html=True)

        # Confusion matrix
        st.markdown("")
        cm_arr = confusion_matrix(
            test_preds.reshape(-1,1) if task=="binary" else test_preds,
            y_te.reshape(-1,1) if task=="binary" else y_te,
            num_classes=n_cls_eval
        )
        class_labels = [str(i) for i in range(n_cls_eval)]
        fig_cm = plot_confusion_matrix(cm_arr, class_labels, title="Test Set Confusion Matrix")
        st.plotly_chart(fig_cm, use_container_width=True)

    else:  # regression
        from utils.loss import mse as mse_fn
        out_tr = model.predict(X_tr)
        out_te = model.predict(X_te)
        
        y_scaler = st.session_state.get("y_scaler")
        if y_scaler is not None:
            out_tr_unscaled = y_scaler.inverse_transform(out_tr)
            out_te_unscaled = y_scaler.inverse_transform(out_te)
            y_te_unscaled = y_scaler.inverse_transform(y_te.reshape(-1, 1)).flatten()
            y_tr_unscaled = y_scaler.inverse_transform(y_tr.reshape(-1, 1)).flatten()
        else:
            out_tr_unscaled = out_tr
            out_te_unscaled = out_te
            y_te_unscaled = y_te
            y_tr_unscaled = y_tr

        rm1, rm2, rm3 = st.columns(3)
        with rm1:
            st.markdown(f'<div class="metric-card"><div class="metric-label">Test MSE</div><div class="metric-value">{mse_metric(out_te_unscaled.ravel(), y_te_unscaled):.2f}</div></div>', unsafe_allow_html=True)
        with rm2:
            st.markdown(f'<div class="metric-card"><div class="metric-label">Test RMSE</div><div class="metric-value">{rmse_metric(out_te_unscaled.ravel(), y_te_unscaled):.2f}</div></div>', unsafe_allow_html=True)
        with rm3:
            st.markdown(f'<div class="metric-card"><div class="metric-label">R² Score</div><div class="metric-value">{r2_score(out_te_unscaled.ravel(), y_te_unscaled):.4f}</div></div>', unsafe_allow_html=True)

        st.markdown('<div class="section-header">📈 True vs Predicted (Test Set)</div>', unsafe_allow_html=True)
        fig_scatter = go.Figure()
        fig_scatter.add_trace(go.Scatter(
            x=y_te_unscaled, y=out_te_unscaled.ravel(),
            mode='markers',
            name='Predictions',
            marker=dict(color='#8b5cf6', size=8, line=dict(color='white', width=1))
        ))
        
        # Identity line
        min_val = min(y_te_unscaled.min(), out_te_unscaled.min())
        max_val = max(y_te_unscaled.max(), out_te_unscaled.max())
        fig_scatter.add_trace(go.Scatter(
            x=[min_val, max_val], y=[min_val, max_val],
            mode='lines',
            name='Ideal Fit',
            line=dict(color='#ef4444', dash='dash')
        ))
        
        fig_scatter.update_layout(
            template='plotly_dark', paper_bgcolor='rgba(15,15,25,1)', plot_bgcolor='rgba(15,15,25,1)',
            xaxis_title="True Values", yaxis_title="Predicted Values",
            height=400, margin=dict(l=40, r=20, t=30, b=40)
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    # ── Decision boundary (only for 2D data) ──
    if X_tr.shape[1] == 2 and task in ["binary", "multiclass"]:
        st.markdown('<div class="section-header">🗺️ Decision Boundary</div>', unsafe_allow_html=True)

        db_col1, db_col2 = st.columns(2)
        with db_col1:
            fig_train_db = plot_decision_boundary(
                model, X_tr, y_tr, title="Training Set — Decision Boundary"
            )
            st.plotly_chart(fig_train_db, use_container_width=True)
        with db_col2:
            fig_test_db = plot_decision_boundary(
                model, X_te, y_te, title="Test Set — Decision Boundary"
            )
            st.plotly_chart(fig_test_db, use_container_width=True)

    # ── Train vs Test bar comparison ──
    if task in ["binary", "multiclass"]:
        st.markdown('<div class="section-header">📈 Train vs Test Accuracy Comparison</div>', unsafe_allow_html=True)
        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(
            x=["Training", "Test"],
            y=[train_acc * 100, test_acc * 100],
            marker=dict(
                color=['rgba(99,102,241,0.8)', 'rgba(168,85,247,0.8)'],
                line=dict(color=['#6366f1', '#a855f7'], width=2),
            ),
            text=[f"{train_acc:.1%}", f"{test_acc:.1%}"],
            textposition='outside',
        ))
        fig_bar.update_layout(
            template='plotly_dark',
            paper_bgcolor='rgba(15,15,25,1)',
            plot_bgcolor='rgba(15,15,25,1)',
            height=300,
            margin=dict(l=30, r=20, t=30, b=40),
            yaxis=dict(range=[0, 115], gridcolor='rgba(255,255,255,0.06)', title="Accuracy (%)"),
        )
        st.plotly_chart(fig_bar, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)
st.markdown(
    '<p style="text-align:center;color:#4a4a6a;font-size:0.8rem;">'
    '🧠 NeuralVis · Built with NumPy + Streamlit · Educational Neural Network Platform'
    '</p>',
    unsafe_allow_html=True
)
