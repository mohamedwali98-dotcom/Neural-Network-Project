# 🧠 NeuralVis: Interactive Neural Network Platform

An interactive educational platform for visualizing and training neural networks, built with Python and Streamlit. This project allows users to experiment with various neural network architectures, hyperparameter tuning, and regularization techniques directly in their browser.

## ✨ Features

* **Multiple Architectures**: Supports Multi-Layer Perceptron (MLP), Modern Perceptron, and Historical Perceptron.
* **Flexible Data Sources**: Train models on synthetic datasets (Make Moons, Make Circles, Make Blobs) or upload your own custom CSV files.
* **Live Visualization**: Watch the training progress in real-time with loss and accuracy curves.
* **Interactive Evaluation**: View decision boundaries (for 2D data), confusion matrices, and detailed evaluation metrics (Accuracy, Precision, Recall, F1, MSE, R²).
* **Comprehensive Hyperparameter Tuning**: Customize learning rates, hidden layers, neurons, activation functions, batch sizes, and regularization (L2, Dropout, Early Stopping).
* **Automated Preprocessing**: Automatically handles scaling (StandardScaler) and label encoding for categorical data when uploading custom CSVs.

## 🚀 Getting Started

### Prerequisites

Ensure you have Python 3 installed.

### Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/mohamedwali98-dotcom/Neural-Network-Project.git
   cd Neural-Network-Project
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the App

To start the Streamlit application, run the following command in your terminal:
```bash
streamlit run app.py
```

The application will automatically open in your default web browser at `http://localhost:8501`.

## 📁 Project Structure

* `app.py`: The main Streamlit application script.
* `models/`: Contains the core implementation of neural network models (`mlp.py`, `perceptron.py`, `layer.py`).
* `training/`: Contains logic for forward/backward propagation and optimizers.
* `reg/`: Implements regularization techniques (L2, Dropout, Early Stopping).
* `utils/`: Helper functions for activations, losses, metrics, and plotting.
