"""
Tabular Deep Learning & Factorization Machine Classifiers.
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


class FactorizationMachineClassifier:
    """
    Order-2 Factorization Machine (FM) for Continuous Pairwise Feature Interactions.
    Reconstructs smooth non-linear interaction manifolds that tree splits miss.
    """
    def __init__(self, k_factors: int = 8, lr: float = 0.01, l2_reg: float = 1e-4, epochs: int = 10, batch_size: int = 4096):
        self.k_factors = k_factors
        self.lr = lr
        self.l2_reg = l2_reg
        self.epochs = epochs
        self.batch_size = batch_size
        self.scaler = StandardScaler()
        self.w0 = 0.0
        self.w = None
        self.V = None
        self.feature_cols = None

    def _sigmoid(self, z: np.ndarray) -> np.ndarray:
        z = np.clip(z, -35.0, 35.0)
        return 1.0 / (1.0 + np.exp(-z))

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        self.feature_cols = list(X.columns)
        X_clean = X.fillna(0.0).values.astype(np.float32)
        X_norm = self.scaler.fit_transform(X_clean)
        n_samples, n_features = X_norm.shape

        rng = np.random.RandomState(42)
        self.w0 = 0.0
        self.w = np.zeros(n_features, dtype=np.float32)
        self.V = rng.normal(scale=0.01, size=(n_features, self.k_factors)).astype(np.float32)

        y_clean = y.astype(np.float32)

        # Vectorized Mini-Batch SGD with Adam-like decay
        for epoch in range(self.epochs):
            indices = np.arange(n_samples)
            rng.shuffle(indices)
            for start_idx in range(0, n_samples, self.batch_size):
                batch_idx = indices[start_idx:start_idx + self.batch_size]
                xb = X_norm[batch_idx]
                yb = y_clean[batch_idx]

                # Linear term: xb @ w
                linear_term = xb @ self.w + self.w0
                # Interaction term: 0.5 * sum((xb @ V)^2 - xb^2 @ V^2)
                xv = xb @ self.V
                xv_sq = xv ** 2
                x_sq_v_sq = (xb ** 2) @ (self.V ** 2)
                interaction_term = 0.5 * np.sum(xv_sq - x_sq_v_sq, axis=1)

                preds = self._sigmoid(linear_term + interaction_term)
                err = preds - yb

                # Gradients
                grad_w0 = np.mean(err)
                grad_w = (xb.T @ err) / len(batch_idx) + self.l2_reg * self.w
                # Vectorized V gradient
                err_col = err[:, np.newaxis, np.newaxis]
                grad_V = (np.transpose(xb[:, :, np.newaxis] * xv[:, np.newaxis, :] - (xb**2)[:, :, np.newaxis] * self.V[np.newaxis, :, :], (1, 2, 0)) @ err[:, np.newaxis]).squeeze(-1) / len(batch_idx) + self.l2_reg * self.V

                # Updates
                self.w0 -= self.lr * grad_w0
                self.w -= self.lr * grad_w
                self.V -= self.lr * grad_V

        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        X_clean = X[self.feature_cols].fillna(0.0).values.astype(np.float32)
        X_norm = self.scaler.transform(X_clean)
        linear_term = X_norm @ self.w + self.w0
        xv = X_norm @ self.V
        interaction_term = 0.5 * np.sum((xv ** 2) - ((X_norm ** 2) @ (self.V ** 2)), axis=1)
        p1 = self._sigmoid(linear_term + interaction_term)
        p0 = 1.0 - p1
        return np.column_stack((p0, p1))
