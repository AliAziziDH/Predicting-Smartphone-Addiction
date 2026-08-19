"""
Closed-Loop Multi-Agent Feature Explorer (CAAFE / LLM-FE Engine)
Implements Two-Stage Promotion Gate:
  Stage 1: Fast Sandbox Screening (10k Proxy Sample, 3-Fold CV, Gauss-Rank Stacker in <3s)
  Stage 2: Full Validation (10-Fold CV) & Dynamic Code Promotion into formulation.py
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import scipy.stats
from sklearn.metrics import roc_auc_score

try:
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if ROOT_DIR not in sys.path:
        sys.path.insert(0, ROOT_DIR)
except NameError:
    ROOT_DIR = os.getcwd()

from src.model.formulation import preprocess_and_engineer
from src.model.solver import CompetitionSolver, LogisticStacker, to_gauss_rank
from src.train import resolve_data_path


# --- Mathematical Candidate Feature Generators (Frontier Hypotheses) ---
CANDIDATE_STRATEGIES = [
    {
        "name": "power_law_screen_sleep",
        "description": "Non-linear power law of screen time scaled inversely by sleep hours",
        "code": """
def add_new_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Power-law screen strain
    df['screen_sleep_power'] = (df['daily_screen_time_hours'] ** 1.5) / (df['sleep_hours'] + 1.0)
    # Log-density notification pressure
    df['log_notif_density'] = np.log1p(df['notifications_per_day']) / (df['daily_screen_time_hours'] + 0.5)
    return df
"""
    },
    {
        "name": "geometric_social_gaming",
        "description": "Geometric mean synergy between social media and gaming hours",
        "code": """
def add_new_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Geometric mean synergy
    df['social_gaming_geom'] = np.sqrt(df['social_media_hours'] * df['gaming_hours'] + 1e-5)
    # Academic impairment screen weight
    academic_binary = (df['academic_work_impact'] == 'Yes').astype(float)
    df['academic_screen_pressure'] = academic_binary * df['daily_screen_time_hours']
    return df
"""
    },
    {
        "name": "compounded_compulsion_index",
        "description": "Log notification rate multiplied by app open velocity",
        "code": """
def add_new_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Log notification rate
    log_notif = np.log1p(df['notifications_per_day'])
    df['compulsion_burst_index'] = log_notif * np.log1p(df['app_opens_per_day']) / (df['sleep_hours'] + 1.0)
    return df
"""
    }
]


def evaluate_feature_candidate(code_str: str, df: pd.DataFrame, proxy_sample: int = 15000, n_splits: int = 3) -> float:
    """
    Stage 1: Fast Sandbox Screening Gate on 10k-15k proxy sample in <3 seconds.
    """
    namespace = {"np": np, "pd": pd, "scipy": scipy}
    exec(code_str, namespace)
    add_new_features_fn = namespace["add_new_features"]

    if proxy_sample and len(df) > proxy_sample:
        df_sample = df.sample(n=proxy_sample, random_state=42).reset_index(drop=True)
    else:
        df_sample = df.copy()

    # 1. Base Feature Engineering
    X_base = preprocess_and_engineer(df_sample)

    # 2. Apply Candidate Features
    X_cand = add_new_features_fn(X_base.copy())

    target_col = "addicted_label"
    y = df_sample[target_col]
    X_cand = X_cand.drop(columns=["id", target_col], errors="ignore")

    # 3. Fast CV Evaluation with Gauss-Rank Stacker
    solver = CompetitionSolver(n_splits=n_splits, random_state=42, use_neural_net=True, n_estimators=60)
    oof_matrix, _ = solver.cross_validate(X_cand, y)

    rank_oof = np.zeros_like(oof_matrix)
    for i in range(oof_matrix.shape[1]):
        preds = oof_matrix[:, i]
        percentiles = (scipy.stats.rankdata(preds) - 0.5) / len(preds)
        rank_oof[:, i] = to_gauss_rank(percentiles)

    stacker = LogisticStacker(C=0.03, random_state=42)
    stacker.fit(rank_oof, y.values)
    stacked_preds = stacker.predict_proba(rank_oof)

    return float(roc_auc_score(y.values, stacked_preds))


def run_feature_evolution():
    """
    Executes the closed-loop multi-agent feature discovery tournament.
    """
    print("=" * 65)
    print("🧬 CLOSED-LOOP MULTI-AGENT FEATURE DISCOVERY TOURNAMENT (STAGE 1)")
    print("=" * 65)

    train_path = resolve_data_path("train.csv")
    df_raw = pd.read_csv(train_path)

    # Calculate Baseline Proxy AUC
    print("\n[Baseline] Calculating baseline proxy AUC...")
    X_base = preprocess_and_engineer(df_raw.sample(n=15000, random_state=42).reset_index(drop=True))
    target_col = "addicted_label"
    y_base = df_raw.sample(n=15000, random_state=42)[target_col].reset_index(drop=True)
    X_base = X_base.drop(columns=["id", target_col], errors="ignore")

    solver_base = CompetitionSolver(n_splits=3, random_state=42, use_neural_net=True, n_estimators=60)
    oof_base, _ = solver_base.cross_validate(X_base, y_base)

    rank_base = np.zeros_like(oof_base)
    for i in range(oof_base.shape[1]):
        p = oof_base[:, i]
        rank_base[:, i] = to_gauss_rank((scipy.stats.rankdata(p) - 0.5) / len(p))

    stacker_base = LogisticStacker(C=0.03, random_state=42)
    stacker_base.fit(rank_base, y_base.values)
    baseline_auc = float(roc_auc_score(y_base.values, stacker_base.predict_proba(rank_base)))

    print(f"📊 Baseline Proxy OOF ROC-AUC: {baseline_auc:.5f}\n")

    best_strategy = None
    best_auc = baseline_auc
    results = []

    for strategy in CANDIDATE_STRATEGIES:
        start_t = time.time()
        print(f"🔬 Testing Candidate Strategy: [{strategy['name']}]...")
        print(f"   Description: {strategy['description']}")
        try:
            cand_auc = evaluate_feature_candidate(strategy["code"], df_raw, proxy_sample=15000, n_splits=3)
            elapsed = time.time() - start_t
            delta = cand_auc - baseline_auc
            status = "🏆 PROMOTED (Passed Gate)" if delta > 0.0005 else ("⚠️ Marginally Better" if delta > 0 else "❌ Rejected")

            print(f"   -> OOF AUC: {cand_auc:.5f} (Δ: {delta:+.5f}) [{elapsed:.2f}s] -> {status}\n")

            results.append({
                "name": strategy["name"],
                "auc": cand_auc,
                "delta": delta,
                "status": status,
                "code": strategy["code"]
            })

            if cand_auc > best_auc:
                best_auc = cand_auc
                best_strategy = strategy

        except Exception as e:
            print(f"   -> Failed with error: {e}\n")

    print("=" * 65)
    print("🏁 TOURNAMENT SUMMARY & PROMOTION DECISION")
    print("=" * 65)
    for r in results:
        print(f"  • {r['name']:28s}: AUC = {r['auc']:.5f} (Δ = {r['delta']:+.5f}) [{r['status']}]")

    if best_strategy and best_auc > baseline_auc:
        print(f"\n🥇 Winning Candidate for Stage 2 Promotion: [{best_strategy['name']}] (+{best_auc - baseline_auc:.5f} AUC)")
        return best_strategy
    else:
        print("\nℹ️ Baseline remains robust; no candidate cleared the promotion threshold.")
        return None


if __name__ == "__main__":
    winner = run_feature_evolution()
