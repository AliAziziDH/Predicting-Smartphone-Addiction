import pytest
import pandas as pd
import numpy as np
import json
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.model.tuner import LeakFreeOptunaTuner

@pytest.fixture
def dummy_data():
    np.random.seed(42)
    n_samples = 1000
    df = pd.DataFrame({
        'age': np.random.randint(10, 80, n_samples),
        'daily_screen_time_hours': np.random.uniform(0, 24, n_samples),
        'social_media_hours': np.random.uniform(0, 10, n_samples),
        'gaming_hours': np.random.uniform(0, 10, n_samples),
        'sleep_hours': np.random.uniform(4, 12, n_samples),
        'notifications_per_day': np.random.poisson(50, n_samples),
        'gender': np.random.choice(['Male', 'Female', 'Other'], n_samples),
        'stress_level': np.random.choice(['Low', 'Medium', 'High'], n_samples),
        'addicted_label': np.random.randint(0, 2, n_samples)
    })
    return df

def test_proxy_downsampling(dummy_data):
    """Assert proxy splits preserve target ratios."""
    X = dummy_data.drop(columns=['addicted_label'])
    y = dummy_data['addicted_label']

    tuner = LeakFreeOptunaTuner(downsample_ratio=0.3)
    X_proxy, y_proxy = tuner._stratified_downsample(X, y, ratio=0.3)

    # Check size
    assert len(X_proxy) == int(0.3 * len(X))
    assert len(y_proxy) == int(0.3 * len(y))

    # Check target distribution
    original_ratio = y.value_counts(normalize=True).sort_index()
    proxy_ratio = y_proxy.value_counts(normalize=True).sort_index()

    # Allow a small tolerance due to rounding in train_test_split
    np.testing.assert_allclose(original_ratio.values, proxy_ratio.values, atol=0.05)


def test_unseen_categories_resilience():
    """Assert local encoding handles unseen categories cleanly."""
    tuner = LeakFreeOptunaTuner(n_splits=2)

    # Create dataset where one fold has a category not present in the other
    X = pd.DataFrame({
        'age': [20, 25, 30, 35, 40],
        'gender': ['Male', 'Female', 'Male', 'Female', 'Other_Unseen'],
        'daily_screen_time_hours': [5.0]*5,
        'social_media_hours': [2.0]*5,
        'gaming_hours': [1.0]*5,
        'sleep_hours': [8.0]*5,
        'notifications_per_day': [10]*5,
        'stress_level': ['Low']*5
    })
    y = pd.Series([0, 1, 0, 1, 0])

    # We will just run the objective function for a small number of trees to verify it doesn't crash
    # Using a dummy trial
    import optuna
    study = optuna.create_study()
    trial = study.ask()

    try:
        # Test if it runs without ValueError for unseen labels
        score = tuner.objective(trial, X, y, model_type='lgb')
        assert isinstance(score, float)
    except Exception as e:
        pytest.fail(f"Tuner objective failed with error: {e}")

def test_config_serialization(tmp_path):
    """Verify optimized parameters are saved as JSON."""
    # We will import the main function from src.tune and mock run_study
    from src.tune import main

    # Mock data_path to a temporary location
    data_path = tmp_path / "data" / "train.csv"
    data_path.parent.mkdir(parents=True)

    # Mock models dir
    models_dir = tmp_path / "models"
    models_dir.mkdir(parents=True)

    with patch('src.tune.Path') as mock_path, \
         patch('src.tune.LeakFreeOptunaTuner.run_study') as mock_run_study:

        # Configure Path mock to return our tmp paths when asked for 'data/train.csv' or 'models'
        def path_side_effect(arg):
            if str(arg) == "data/train.csv":
                return data_path
            elif str(arg) == "models":
                return models_dir
            return Path(arg)

        mock_path.side_effect = path_side_effect

        # Mock run_study to return a dummy config
        def run_study_side_effect(X, y, model_type, n_trials):
            if model_type == 'lgb':
                return {'n_estimators': 100, 'learning_rate': 0.1}
            elif model_type == 'xgb':
                return {'n_estimators': 200, 'learning_rate': 0.05}
            elif model_type == 'cat':
                return {'iterations': 300, 'learning_rate': 0.01}
            return {}

        mock_run_study.side_effect = run_study_side_effect

        # Run main
        main()

        # Verify JSON was created
        output_json = models_dir / "best_hyperparameters.json"
        assert output_json.exists()

        # Verify JSON contents
        with open(output_json, 'r') as f:
            saved_config = json.load(f)

        assert saved_config['lgb'] == {'n_estimators': 100, 'learning_rate': 0.1}
        assert saved_config['xgb'] == {'n_estimators': 200, 'learning_rate': 0.05}
        assert saved_config['cat'] == {'iterations': 300, 'learning_rate': 0.01}
