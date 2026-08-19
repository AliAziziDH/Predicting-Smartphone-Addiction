"""
Wave 4/5: Autonomous Multi-Generation Feature Discovery Engine powered by Google AI Studio (Gemini 3.1 Pro).
Directly leverages flagship Gemini 3.1 Pro via StudioEngine,
evaluates proposals in <5s on 15k proxy rows with Gauss-Rank Stacking,
and logs exact token economy and feature promotion gates.
"""

import os
import sys
import time
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata

try:
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if ROOT_DIR not in sys.path:
        sys.path.insert(0, ROOT_DIR)
except NameError:
    ROOT_DIR = os.getcwd()

from src.model.formulation import preprocess_and_engineer
from src.model.solver import CompetitionSolver, to_gauss_rank, LogisticStacker
from src.studio_engine import get_studio_engine, StudioEngine
from src.train import resolve_data_path


def query_studio_for_candidates(
    generation: int,
    current_best_auc: float,
    failed_candidates: List[Dict[str, Any]],
    winning_candidates: List[Dict[str, Any]],
    engine: StudioEngine,
) -> List[Dict[str, str]]:
    """
    Directly queries Gemini 3.1 Pro via StudioEngine for high-yield mathematical feature candidates.
    """
    failed_summary = "\n".join([f"- {f['name']} (Δ = {f['delta']:+.5f}): {f['description']}" for f in failed_candidates[-4:]]) or "None yet."
    winning_summary = "\n".join([f"- {w['name']} (AUC = {w['auc']:.5f}): {w['description']}" for w in winning_candidates]) or "Baseline (productive_work_shield, work_adjusted_screen_load, unaccounted_hours) -> AUC: 0.93437"

    system_instruction = (
        "You are Principal AI Decision Intelligence Architect & Kaggle Grandmaster.\n"
        "Your task is to design novel non-linear, ratio-based, and interaction features for Tabular GBDT.\n"
        "You MUST return valid JSON containing a list of feature candidate functions."
    )

    prompt = f"""We are optimizing Out-of-Fold ROC-AUC for Kaggle Playground S6E8 (Smartphone Addiction).
Current Best Proxy OOF AUC: {current_best_auc:.5f}

--- WINNING FEATURE COMBINATIONS ---
{winning_summary}

--- FAILED RECENT EXPERIMENTS (AVOID THESE TRAPS) ---
{failed_summary}

--- AVAILABLE BASE COLUMNS & TYPES ---
- Numeric: age (int), daily_screen_time_hours (float), sleep_hours (float), social_media_hours (float), gaming_hours (float), work_study_hours (float), app_opens_per_day (float), notifications_per_day (float), weekend_screen_time (float), other_screen (float), unaccounted_hours (float), productive_work_ratio (float), work_adjusted_screen_load (float)
- Categorical/String: gender ('Male'/'Female'), stress_level ('Low'/'Medium'/'High'), academic_work_impact ('No'/'Yes')
NOTE: If using stress_level or academic_work_impact in math, map them first e.g.:
`stress_map = {'Low': 1, 'Medium': 2, 'High': 3}; s = df['stress_level'].map(stress_map).fillna(2)`

REQUIREMENTS:
1. Propose exactly 3 innovative, non-linear Python feature engineering functions.
2. Avoid simple monotonic powers or linear sums that GBDT trees already learn.
3. Target:
   - Compulsive frequency densities (e.g. notifications per awake minute)
   - Cross-cohort residuals (e.g. gaming vs group-normalized productive ratio)
   - Sleep debt and circadian disruption interaction penalties
   - Boundary distance non-linear transforms (e.g. |screen_time - 5.5| / awake_time)
4. Output MUST be strictly valid JSON without preamble:
{{
  "candidates": [
    {{
      "name": "feature_name_in_snake_case",
      "description": "Brief mathematical rationale",
      "code": "def add_new_features(df: pd.DataFrame) -> pd.DataFrame:\\n    df = df.copy()\\n    # your engineered features here\\n    return df"
    }}
  ]
}}"""

    try:
        parsed = engine.generate_json(
            prompt=prompt,
            system_instruction=system_instruction,
            temperature=0.2
        )
        if isinstance(parsed, dict) and "candidates" in parsed:
            return parsed["candidates"]
        elif isinstance(parsed, list):
            return parsed
        return []
    except Exception as e:
        print(f"⚠️ [FeatureExplorer] Studio query error: {e}")
        return []


def evaluate_feature_candidate(
    feature_code: str,
    df: pd.DataFrame,
    proxy_sample: int = 15000,
    n_splits: int = 3
) -> float:
    """Fast-Screening Sandbox Gate: Evaluates candidate code in <5 seconds."""
    df_sample = df.sample(n=min(proxy_sample, len(df)), random_state=42).reset_index(drop=True)

    local_scope: Dict[str, Any] = {"pd": pd, "np": np}
    exec(feature_code, local_scope)
    add_new_features_fn = local_scope.get("add_new_features")
    if not add_new_features_fn:
        raise ValueError("Candidate code does not define `add_new_features` function.")

    X_base = preprocess_and_engineer(df_sample)
    X_cand = add_new_features_fn(X_base.copy())

    target_col = "addicted_label"
    y = df_sample[target_col]
    X_cand = X_cand.drop(columns=["id", target_col], errors="ignore")

    solver = CompetitionSolver(n_splits=n_splits, random_state=42, use_neural_net=False, n_estimators=50)
    oof_matrix, _ = solver.cross_validate(X_cand, y)

    rank_matrix = np.zeros_like(oof_matrix)
    for i in range(oof_matrix.shape[1]):
        r = (rankdata(oof_matrix[:, i]) - 0.5) / len(oof_matrix)
        rank_matrix[:, i] = to_gauss_rank(r)

    stacker = LogisticStacker(C=0.03, random_state=42)
    stacker.fit(rank_matrix, y.values)
    final_oof_probs = stacker.predict_proba(rank_matrix)

    score = float(roc_auc_score(y.values, final_oof_probs))
    return score


def run_evolutionary_search(n_generations: int = 2, proxy_sample: int = 15000):
    print("=" * 70)
    print("🧬 AUTONOMOUS FEATURE SEARCH ENGINE (GOOGLE AI STUDIO - GEMINI 3.1 PRO)")
    print("=" * 70)

    train_path = resolve_data_path("train.csv")
    df = pd.read_csv(train_path)

    engine = get_studio_engine()

    # 1. Baseline Evaluation
    print("\n[Generation 0: Baseline] Evaluating Current Feature Space...")
    df_sample = df.sample(n=min(proxy_sample, len(df)), random_state=42).reset_index(drop=True)
    target_col = "addicted_label"
    y_base = df_sample[target_col]
    X_base = df_sample.drop(columns=["id", target_col], errors="ignore")

    solver_base = CompetitionSolver(n_splits=3, random_state=42, use_neural_net=False, n_estimators=50)
    oof_base, _ = solver_base.cross_validate(X_base, y_base)

    rank_base = np.zeros_like(oof_base)
    for i in range(oof_base.shape[1]):
        r = (rankdata(oof_base[:, i]) - 0.5) / len(oof_base)
        rank_base[:, i] = to_gauss_rank(r)

    stacker_base = LogisticStacker(C=0.03, random_state=42)
    stacker_base.fit(rank_base, y_base.values)
    baseline_auc = float(roc_auc_score(y_base.values, stacker_base.predict_proba(rank_base)))

    print(f"📊 Current Baseline Proxy OOF ROC-AUC: {baseline_auc:.5f}\n")

    current_best_auc = baseline_auc
    winning_features: List[Dict[str, Any]] = []
    failed_features: List[Dict[str, Any]] = []

    for gen in range(1, n_generations + 1):
        print(f"\n{'='*25} GENERATION {gen}/{n_generations} {'='*25}")

        candidates = query_studio_for_candidates(
            generation=gen,
            current_best_auc=current_best_auc,
            failed_candidates=failed_features,
            winning_candidates=winning_features,
            engine=engine,
        )

        if not candidates:
            print("  ⚠️ No candidates received from Google AI Studio in this round.")
            continue

        gen_best_cand = None
        gen_best_auc = current_best_auc

        for cand in candidates:
            cand_name = cand.get("name", "unknown")
            desc = cand.get("description", "")
            print(f"  🔬 Testing AI Studio Proposal: [{cand_name}] - {desc}...")
            start_t = time.time()
            try:
                score = evaluate_feature_candidate(cand["code"], df, proxy_sample=proxy_sample, n_splits=3)
                elapsed = time.time() - start_t
                delta = score - current_best_auc

                if delta >= 0.0003:
                    status = "🏆 PROMOTED (Passed Gate)"
                    if score > gen_best_auc:
                        gen_best_auc = score
                        gen_best_cand = cand
                else:
                    status = "❌ Rejected (Below threshold)"
                    failed_features.append({
                        "name": cand_name,
                        "description": desc,
                        "delta": delta
                    })

                print(f"     -> OOF AUC: {score:.5f} (Δ: {delta:+.5f}) [{elapsed:.2f}s] -> {status}")
            except Exception as e:
                print(f"     -> Error during evaluation: {e}")
                failed_features.append({
                    "name": cand_name,
                    "description": f"Syntax/Runtime Error: {e}",
                    "delta": -0.01
                })

        if gen_best_cand is not None:
            print(f"\n  ✨ Generation {gen} Winner: [{gen_best_cand['name']}] (AUC: {gen_best_auc:.5f})")
            current_best_auc = gen_best_auc
            winning_features.append({
                "generation": gen,
                "name": gen_best_cand["name"],
                "description": gen_best_cand.get("description", ""),
                "auc": gen_best_auc,
                "delta": gen_best_auc - baseline_auc,
                "code": gen_best_cand.get("code", "")
            })
        else:
            print(f"\n  ℹ️ Generation {gen}: No candidate cleared promotion threshold. Baseline retained.")

    print("\n" + "=" * 70)
    print("🏁 LIVE EVOLUTIONARY TOURNAMENT SUMMARY & GOOGLE AI STUDIO REPORT")
    print("=" * 70)
    print(f"📊 Initial Baseline AUC: {baseline_auc:.5f}")
    print(f"🏆 Final Best AUC:       {current_best_auc:.5f} (Total Uplift: {current_best_auc - baseline_auc:+.5f})")
    print(f"\n{engine.get_usage_summary()}")
    print("=" * 70)

    return {
        "baseline_auc": baseline_auc,
        "best_auc": current_best_auc,
        "winning_features": winning_features
    }


if __name__ == "__main__":
    run_evolutionary_search(n_generations=2, proxy_sample=15000)
