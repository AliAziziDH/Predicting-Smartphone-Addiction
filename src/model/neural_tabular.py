"""
Tabular Multi-Layer Perceptron (Neural Network) Classifier.
Provides non-tree continuous inductive bias for maximum ensemble diversity.
Optimized for high-speed multi-threaded CPU execution with zero CUDA binary dependencies.
"""
import numpy as np
import pandas as pd
from typing import List, Optional
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

class DeepTabularClassifier:
    """
    Scikit-Learn compatible Neural Network Classifier for Tabular Data.
    Features robust standardization, early stopping, and fast batch execution.
    """
    def __init__(
        self,
        hidden_dim: int = 128,
        num_blocks: int = 2,
        dropout: float = 0.15,
        lr: float = 2e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 8192,
        epochs: int = 12,
        device: Optional[str] = None
    ):
        self.hidden_dim = hidden_dim
        self.num_blocks = num_blocks
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.epochs = epochs

        hidden_layers = (self.hidden_dim, self.hidden_dim // 2) if num_blocks > 1 else (self.hidden_dim,)
        self.model = MLPClassifier(
            hidden_layer_sizes=hidden_layers,
            activation='relu',
            solver='adam',
            alpha=self.weight_decay,
            batch_size=self.batch_size,
            learning_rate_init=self.lr,
            max_iter=self.epochs,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=3,
            random_state=42
        )
        self.scaler = StandardScaler()
        self.feature_cols = None

    def fit(self, X: pd.DataFrame, y: np.ndarray, cat_cols: Optional[List[str]] = None):
        self.feature_cols = list(X.columns)
        X_clean = X.fillna(0.0).values.astype(np.float32)
        X_scaled = self.scaler.fit_transform(X_clean)
        self.model.fit(X_scaled, y)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        X_clean = X[self.feature_cols].fillna(0.0).values.astype(np.float32)
        X_scaled = self.scaler.transform(X_clean)
        return self.model.predict_proba(X_scaled)
