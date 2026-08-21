import os
import sys

# Dynamic path resolution to handle running from subfolders or root
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if os.path.basename(ROOT_DIR) in ["src", "tests"]:
    ROOT_DIR = os.path.dirname(ROOT_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

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

def test_nelder_mead_rank_stacker(mock_predictions):
    from src.model.solver import NelderMeadRankStacker
    import scipy.stats

    preds_matrix, y = mock_predictions
    rank_matrix = np.zeros_like(preds_matrix)
    for i in range(preds_matrix.shape[1]):
        rank_matrix[:, i] = (scipy.stats.rankdata(preds_matrix[:, i]) - 0.5) / len(preds_matrix)

    stacker = NelderMeadRankStacker(random_state=42)
    stacker.fit(rank_matrix, y)

    assert stacker.weights_ is not None
    assert np.all(stacker.weights_ >= 0.0)
    assert np.isclose(np.sum(stacker.weights_), 1.0, atol=1e-4)

    blended_probs = stacker.predict_proba(rank_matrix)
    stacker_auc = roc_auc_score(y, blended_probs)

    auc_lgb = roc_auc_score(y, rank_matrix[:, 0])
    auc_xgb = roc_auc_score(y, rank_matrix[:, 1])
    auc_cat = roc_auc_score(y, rank_matrix[:, 2])
    best_single_auc = max(auc_lgb, auc_xgb, auc_cat)

    assert stacker_auc >= best_single_auc - 1e-6, f"Nelder-Mead AUC ({stacker_auc:.5f}) should be >= best single model ({best_single_auc:.5f})"


def test_two_stage_hybrid_stacker_old(mock_predictions):
    from src.model.solver import EnsembleBlender
    preds_matrix, y = mock_predictions
    p_lgb = preds_matrix[:, 0]
    p_xgb = preds_matrix[:, 1]
    p_cat = preds_matrix[:, 2]
    # Synthetic MLP predictions
    np.random.seed(42)
    p_nn = np.clip(y * 0.5 + np.random.normal(0.4, 0.25, len(y)), 0.01, 0.99)

    stacker = EnsembleBlender(random_state=42)
    stacker.fit(p_lgb, p_cat, p_xgb, p_nn, y)

    assert stacker.tree_weights_ is not None
    assert np.isclose(np.sum(stacker.tree_weights_), 1.0, atol=1e-4)
    assert 0.0 <= stacker.alpha_ <= 1.0

    preds_final = stacker.predict_proba(p_lgb, p_cat, p_xgb, p_nn)
    final_auc = roc_auc_score(y, preds_final)
    best_single = max(roc_auc_score(y, p_lgb), roc_auc_score(y, p_xgb), roc_auc_score(y, p_cat))

    assert final_auc >= best_single - 1e-5


