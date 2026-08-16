import pytest
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from src.model.solver import EnsembleBlender

@pytest.fixture
def mock_predictions():
    np.random.seed(42)
    n_samples = 1000

    # Generate true labels
    y = np.random.randint(0, 2, n_samples)

    # Generate predictions that are somewhat correlated with the target
    preds_lgb = np.clip(y * 0.7 + np.random.normal(0.2, 0.2, n_samples), 0, 1)
    preds_xgb = np.clip(y * 0.6 + np.random.normal(0.3, 0.2, n_samples), 0, 1)

    # Let's make catboost a bit worse to see if weights adjust
    preds_cat = np.clip(y * 0.4 + np.random.normal(0.5, 0.3, n_samples), 0, 1)

    preds_matrix = np.column_stack((preds_lgb, preds_xgb, preds_cat))
    return preds_matrix, y

def test_blend_weights_constraints(mock_predictions):
    preds_matrix, y = mock_predictions
    blender = EnsembleBlender()

    weights = blender.fit(preds_matrix, y)

    # Constraint a: Non-negativity constraint [0.0, 1.0]
    assert np.all(weights >= -1e-6), f"Weights contain negative values: {weights}"
    assert np.all(weights <= 1.0 + 1e-6), f"Weights contain values > 1.0: {weights}"

    # Constraint b: Sum-to-one constraint
    assert np.isclose(np.sum(weights), 1.0, atol=1e-5), f"Weights do not sum to exactly 1.0. Sum: {np.sum(weights)}"

def test_blending_auc_improvement(mock_predictions):
    preds_matrix, y = mock_predictions
    blender = EnsembleBlender()

    weights = blender.fit(preds_matrix, y)

    # Calculate baseline AUCs
    auc_lgb = roc_auc_score(y, preds_matrix[:, 0])
    auc_xgb = roc_auc_score(y, preds_matrix[:, 1])
    auc_cat = roc_auc_score(y, preds_matrix[:, 2])

    best_baseline_auc = max(auc_lgb, auc_xgb, auc_cat)

    # Calculate Optimized OOF AUC
    optimized_preds = np.dot(preds_matrix, weights)
    optimized_auc = roc_auc_score(y, optimized_preds)

    # The optimized blend should be at least as good as the best individual model
    # (Allowing a very small numerical tolerance)
    assert optimized_auc >= best_baseline_auc - 1e-6, \
        f"Optimized AUC ({optimized_auc:.6f}) is worse than best baseline ({best_baseline_auc:.6f})"
