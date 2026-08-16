#!/usr/bin/env python3
import json
import logging
import os
import numpy as np
from pathlib import Path

import pandas as pd

from src.model.tuner import LeakFreeOptunaTuner

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    logger.info("Starting hyperparameter tuning process...")

    # Load dataset
    data_path = Path("data/train.csv")
    if not data_path.exists():
        logger.warning(f"Dataset not found at {data_path}. Generating dummy data for testing purposes.")
        os.makedirs("data", exist_ok=True)
        # Create a small realistic dataset
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
        df.to_csv(data_path, index=False)

    logger.info(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)

    if 'addicted_label' not in df.columns:
        raise ValueError("Target column 'addicted_label' not found in dataset.")

    X = df.drop(columns=['addicted_label'])
    y = df['addicted_label']

    # Instantiate the tuner
    tuner = LeakFreeOptunaTuner(n_splits=3, random_state=42, downsample_ratio=0.3)

    best_params_all = {}

    models_to_tune = ['lgb', 'xgb', 'cat']
    n_trials = 20

    for model_type in models_to_tune:
        try:
            best_params = tuner.run_study(X, y, model_type=model_type, n_trials=n_trials)
            best_params_all[model_type] = best_params
            logger.info(f"Successfully tuned {model_type}")
        except Exception as e:
            logger.error(f"Failed to tune {model_type}: {e}")

    # Save the optimal hyperparameters
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)

    output_path = models_dir / "best_hyperparameters.json"
    with open(output_path, "w") as f:
        json.dump(best_params_all, f, indent=4)

    logger.info(f"Best hyperparameters saved to {output_path}")

if __name__ == "__main__":
    main()
