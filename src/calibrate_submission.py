"""
Probability & Rank Percentile Calibrator for Submission Files.
Guarantees strict [0.0, 1.0] probability bounds for Kaggle AUC evaluations.
"""

import pandas as pd
import numpy as np
from scipy.stats import norm, rankdata


def calibrate_submission(input_path: str = "submission_elite_wave8.csv", output_path: str = "submission_elite_wave8_calibrated.csv"):
    df = pd.read_csv(input_path)
    target_col = [c for c in df.columns if c != "id"][0]
    raw_vals = df[target_col].values

    # 1. If values are Gaussian rank projections, map through Standard Normal CDF to [0, 1]
    if np.min(raw_vals) < 0 or np.max(raw_vals) > 1.0:
        print(f"• Input values range from [{np.min(raw_vals):.4f}, {np.max(raw_vals):.4f}]. Converting to strict [0.0, 1.0] probabilities via Normal CDF & Rank Projection...")
        prob_vals = norm.cdf(raw_vals)
    else:
        prob_vals = raw_vals

    # Ensure strictly in [0.0, 1.0]
    prob_vals = np.clip(prob_vals, 0.0, 1.0)
    
    df[target_col] = prob_vals
    df.to_csv(output_path, index=False)
    print(f"✅ Calibrated submission saved to: {output_path}")
    print(f"• Min: {np.min(prob_vals):.5f} | Max: {np.max(prob_vals):.5f} | Mean: {np.mean(prob_vals):.5f} | Shape: {df.shape}")
    
    # Also overwrite the primary submission file
    df.to_csv(input_path, index=False)
    print(f"✅ Primary submission '{input_path}' updated with strict [0.0, 1.0] probabilities.")


if __name__ == "__main__":
    calibrate_submission()
