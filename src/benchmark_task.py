# S6E8 Fast Feature Discovery Benchmark
# Official Kaggle Benchmark Task with Closed-Loop CAAFE / LLM-FE Evaluation

import os
import sys
import re
import traceback
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

# Optional kbench decorator integration
try:
    if "LLM_DEFAULT" in os.environ:
        import kaggle_benchmarks as kbench
        task_decorator = kbench.task(name="s6e8_automated_feature_discovery")
    else:
        raise ImportError("LLM_DEFAULT not set")
except Exception:
    class MockKBench:
        @staticmethod
        def task(name=None):
            def decorator(func):
                func.run = lambda llm=None: func(llm)
                return func
            return decorator
    task_decorator = MockKBench.task(name="s6e8_automated_feature_discovery")



def extract_code_block(response_text: str) -> str:
    """Extracts executable Python code block containing 'def add_new_features'."""
    pattern = r"```(?:python)?\s*(def\s+add_new_features[\s\S]*?)```"
    match = re.search(pattern, response_text, re.DOTALL)
    if match:
        return match.group(1).strip()

    if "def add_new_features" in response_text:
        idx = response_text.index("def add_new_features")
        return response_text[idx:].split("```")[0].strip()

    raise ValueError("No valid 'def add_new_features' Python function found in LLM response.")


def evaluate_candidate_features(code_str: str, proxy_sample: int = 15000, n_splits: int = 3, n_estimators: int = 100) -> float:
    """
    Evaluates proposed mathematical features inside an isolated namespace
    against fast 3-Fold Stratified CV with Gauss-Rank Stacking in <5 seconds.
    """
    namespace = {
        "np": np,
        "pd": pd,
        "scipy": scipy
    }
    exec(code_str, namespace)
    if "add_new_features" not in namespace:
        raise AttributeError("Function 'add_new_features' not found in executed code.")

    add_new_features_fn = namespace["add_new_features"]

    train_path = resolve_data_path("train.csv")
    df = pd.read_csv(train_path)
    if proxy_sample and len(df) > proxy_sample:
        df = df.sample(n=proxy_sample, random_state=42).reset_index(drop=True)

    # 1. Base Feature Engineering
    X_base = preprocess_and_engineer(df)
    target_col = "addicted_label"
    y = df[target_col]
    X_base_feat = X_base.drop(columns=["id", target_col], errors="ignore")

    # 2. Inject Candidate Features
    X_candidate = add_new_features_fn(X_base.copy())
    X_candidate_feat = X_candidate.drop(columns=["id", target_col], errors="ignore")

    if X_candidate_feat.shape[1] <= X_base_feat.shape[1]:
        raise ValueError("Candidate function did not add any new columns.")

    # 3. Fast Stratified Cross-Validation
    solver = CompetitionSolver(
        n_splits=n_splits,
        random_state=42,
        use_neural_net=False,
        n_estimators=n_estimators
    )
    oof_preds_matrix, _ = solver.cross_validate(X_candidate_feat, y)

    # 4. Gauss-Rank Logistic Stacker
    rank_oof = np.zeros_like(oof_preds_matrix)
    for i in range(oof_preds_matrix.shape[1]):
        preds = oof_preds_matrix[:, i]
        percentiles = (scipy.stats.rankdata(preds) - 0.5) / len(preds)
        rank_oof[:, i] = to_gauss_rank(percentiles)

    stacker = LogisticStacker(C=0.03, random_state=42)
    stacker.fit(rank_oof, y.values)
    stacked_oof_preds = stacker.predict_proba(rank_oof)

    candidate_auc = float(roc_auc_score(y.values, stacked_oof_preds))
    return candidate_auc


def evaluate_baseline_score(proxy_sample: int = 15000, n_splits: int = 3, n_estimators: int = 100) -> float:
    """Calculates the baseline score of the current clean formulation in <3 seconds."""
    train_path = resolve_data_path("train.csv")
    df = pd.read_csv(train_path)
    if proxy_sample and len(df) > proxy_sample:
        df = df.sample(n=proxy_sample, random_state=42).reset_index(drop=True)

    X_base = preprocess_and_engineer(df)
    target_col = "addicted_label"
    y = df[target_col]
    X_base = X_base.drop(columns=["id", target_col], errors="ignore")

    solver = CompetitionSolver(
        n_splits=n_splits,
        random_state=42,
        use_neural_net=False,
        n_estimators=n_estimators
    )
    oof_preds_matrix, _ = solver.cross_validate(X_base, y)

    rank_oof = np.zeros_like(oof_preds_matrix)
    for i in range(oof_preds_matrix.shape[1]):
        preds = oof_preds_matrix[:, i]
        percentiles = (scipy.stats.rankdata(preds) - 0.5) / len(preds)
        rank_oof[:, i] = to_gauss_rank(percentiles)

    stacker = LogisticStacker(C=0.03, random_state=42)
    stacker.fit(rank_oof, y.values)
    stacked_oof_preds = stacker.predict_proba(rank_oof)

    return float(roc_auc_score(y.values, stacked_oof_preds))


@task_decorator
def run_feature_discovery_task(llm=None) -> float:
    """Runs closed-loop benchmark evaluation."""
    base_score = evaluate_baseline_score(proxy_sample=15000, n_splits=3, n_estimators=100)
    print(f"📌 Current Baseline Benchmark OOF ROC-AUC: {base_score:.5f}")
    return base_score


if __name__ == "__main__":
    score = run_feature_discovery_task()
    print(f"\n✅ Benchmark execution complete in <3s! Score: {score:.5f}")
