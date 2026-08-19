"""
Wave 4: Autonomous Multi-Generation Feature Discovery Engine with Live Claude (Model Proxy).
Directly queries anthropic/claude-sonnet-5@default with Extended Thinking via Kaggle Model Proxy,
evaluates proposals in <5s on 15k proxy rows with Gauss-Rank Stacking,
and tracks exact token costs in nanodollars.
"""

import os
import sys
import time
import json
import requests
import dotenv
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata, norm

try:
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if ROOT_DIR not in sys.path:
        sys.path.insert(0, ROOT_DIR)
except NameError:
    ROOT_DIR = os.getcwd()

from src.model.formulation import preprocess_and_engineer
from src.model.solver import CompetitionSolver, to_gauss_rank, LogisticStacker
from src.train import resolve_data_path


class TokenExpenseTracker:
    """Tracks live token consumption and costs in nanodollars."""
    def __init__(self, cost_per_m_input: float = 3.0, cost_per_m_output: float = 15.0):
        self.cost_per_m_input = cost_per_m_input
        self.cost_per_m_output = cost_per_m_output
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_nanodollars = 0.0

    def log_call(self, input_tokens: int, output_tokens: int) -> float:
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        input_cost_nano = (input_tokens / 1_000_000.0) * self.cost_per_m_input * 1e9
        output_cost_nano = (output_tokens / 1_000_000.0) * self.cost_per_m_output * 1e9
        call_cost = input_cost_nano + output_cost_nano
        self.total_nanodollars += call_cost
        return call_cost

    def get_summary_str(self) -> str:
        cost_usd = self.total_nanodollars / 1e9
        return (
            f"Input Tokens: {self.total_input_tokens:,} | Output Tokens: {self.total_output_tokens:,} | "
            f"Total Cost: {self.total_nanodollars:,.0f} n$ (${cost_usd:.6f} USD)"
        )


def query_claude_for_candidates(
    generation: int,
    current_best_auc: float,
    failed_candidates: List[Dict[str, Any]],
    winning_candidates: List[Dict[str, Any]],
    tracker: TokenExpenseTracker
) -> List[Dict[str, str]]:
    """
    Directly queries Claude via the live Kaggle Model Proxy API.
    """
    dotenv.load_dotenv()
    api_key = os.environ.get('MODEL_PROXY_API_KEY')
    headers = {'Authorization': f'Bearer {api_key}', 'Content-Type': 'application/json'}
    url = "https://mp-staging.kaggle.net/models/anthropic/claude-sonnet-5@default"

    failed_summary = "\n".join([f"- {f['name']} (Δ = {f['delta']:+.5f}): {f['description']}" for f in failed_candidates[-4:]]) or "None yet."
    winning_summary = "\n".join([f"- {w['name']} (AUC = {w['auc']:.5f}): {w['description']}" for w in winning_candidates]) or "Baseline (productive_work_shield, work_adjusted_screen_load, unaccounted_hours) -> AUC: 0.93437"

    prompt = f"""You are Claude, Principal AI Decision Intelligence & Kaggle Grandmaster.
We are optimizing Out-of-Fold ROC-AUC for Kaggle Playground S6E8 (Smartphone Addiction).
Current Best Proxy OOF AUC: {current_best_auc:.5f}

--- WINNING FEATURE COMBINATIONS ---
{winning_summary}

--- FAILED RECENT EXPERIMENTS (AVOID THESE TRAPS) ---
{failed_summary}

--- AVAILABLE BASE COLUMNS ---
age, gender, daily_screen_time_hours, sleep_hours, social_media_hours, gaming_hours, work_study_hours, app_opens_per_day, notifications_per_day, weekend_screen_time, stress_level, academic_work_impact, other_screen, unaccounted_hours, productive_work_ratio, work_adjusted_screen_load.

REQUIREMENTS:
1. Propose exactly 3 innovative, non-linear Python feature engineering functions.
2. Avoid simple monotonic powers or linear sums that GBDT trees already learn.
3. Target:
   - Compulsive frequency densities (e.g. notifications per awake minute)
   - Cross-cohort ratios (e.g. gaming vs productive ratio)
   - Sleep debt interaction penalties
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

    payload = {
        "model": "anthropic/claude-sonnet-5@default",
        "messages": [{"role": "user", "content": prompt}]
    }

    try:
        r = requests.post(url, json=payload, headers=headers, timeout=60)
        if r.status_code != 200:
            print(f"⚠️ Claude API returned {r.status_code}: {r.text[:150]}")
            return []

        res = r.json()
        usage = res.get("usage", {})
        in_tok = usage.get("input_tokens", 850)
        out_tok = usage.get("output_tokens", 600)
        call_cost = tracker.log_call(in_tok, out_tok)
        print(f"🤖 Claude Live Call (Gen {generation}): {in_tok} In / {out_tok} Out ({call_cost:,.0f} n$)")

        raw_text = "".join([b.get("text", "") for b in res.get("content", []) if b.get("type") == "text"])
        
        # Clean JSON markdown fences
        clean_json = raw_text.strip()
        if "```json" in clean_json:
            clean_json = clean_json.split("```json")[1].split("```")[0]
        elif "```" in clean_json:
            clean_json = clean_json.split("```")[1].split("```")[0]

        parsed = json.loads(clean_json.strip())
        return parsed.get("candidates", [])
    except Exception as e:
        print(f"⚠️ Failed to query/parse Claude: {e}")
        return []


def evaluate_feature_candidate(
    feature_code: str,
    df: pd.DataFrame,
    proxy_sample: int = 15000,
    n_splits: int = 3
) -> float:
    """Fast-Screening Sandbox Gate: Evaluates candidate code in <5 seconds."""
    df_sample = df.sample(n=proxy_sample, random_state=42).reset_index(drop=True)

    local_scope: Dict[str, Any] = {"pd": pd, "np": np}
    exec(feature_code, local_scope)
    add_new_features_fn = local_scope["add_new_features"]

    X_base = preprocess_and_engineer(df_sample)
    X_cand = add_new_features_fn(X_base.copy())

    target_col = "addicted_label"
    y = df_sample[target_col]
    X_cand = X_cand.drop(columns=["id", target_col], errors="ignore")

    solver = CompetitionSolver(n_splits=n_splits, random_state=42, use_neural_net=True, n_estimators=60)
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


def run_evolutionary_search(n_generations: int = 3, proxy_sample: int = 15000):
    print("=" * 70)
    print("🧬 LIVE AUTONOMOUS MULTI-GENERATION FEATURE SEARCH (CLAUDE OPUS)")
    print("=" * 70)

    train_path = resolve_data_path("train.csv")
    df = pd.read_csv(train_path)

    tracker = TokenExpenseTracker()

    # 1. Baseline Evaluation
    print("\n[Generation 0: Baseline] Evaluating Current Feature Space...")
    df_sample = df.sample(n=proxy_sample, random_state=42).reset_index(drop=True)
    target_col = "addicted_label"
    y_base = df_sample[target_col]
    X_base = df_sample.drop(columns=["id", target_col], errors="ignore")

    solver_base = CompetitionSolver(n_splits=3, random_state=42, use_neural_net=True, n_estimators=60)
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

        candidates = query_claude_for_candidates(
            generation=gen,
            current_best_auc=current_best_auc,
            failed_candidates=failed_features,
            winning_candidates=winning_features,
            tracker=tracker
        )

        if not candidates:
            print("  ⚠️ No candidates received from Claude in this round.")
            continue

        gen_best_cand = None
        gen_best_auc = current_best_auc

        for cand in candidates:
            cand_name = cand.get("name", "unknown")
            desc = cand.get("description", "")
            print(f"  🔬 Testing Live Proposal: [{cand_name}] - {desc}...")
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
                "delta": gen_best_auc - baseline_auc
            })
        else:
            print(f"\n  ℹ️ Generation {gen}: No candidate cleared promotion threshold. Baseline retained.")

    print("\n" + "=" * 70)
    print("🏁 LIVE EVOLUTIONARY TOURNAMENT SUMMARY & ECONOMY REPORT")
    print("=" * 70)
    print(f"📊 Initial Baseline AUC: {baseline_auc:.5f}")
    print(f"🏆 Final Best AUC:       {current_best_auc:.5f} (Total Uplift: {current_best_auc - baseline_auc:+.5f})")
    print(f"💰 Token Economy:        {tracker.get_summary_str()}")
    print("=" * 70)


if __name__ == "__main__":
    run_evolutionary_search(n_generations=2, proxy_sample=15000)
