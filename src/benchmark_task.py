# %% [markdown]
# # S6E8 Automated Multi-Agent Feature Discovery Benchmark
# Official Kaggle Benchmark Task with Closed-Loop CAAFE / LLM-FE Evaluation

# %%
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
    import kaggle_benchmarks as kbench
    task_decorator = kbench.task(name="s6e8_automated_feature_discovery")
except (ImportError, AttributeError):
    try:
        import kbench
        task_decorator = kbench.task(name="s6e8_automated_feature_discovery")
    except (ImportError, AttributeError):
        class MockKBench:
            @staticmethod
            def task(name=None):
                def decorator(func):
                    func.run = lambda llm=None: func(llm)
                    return func
                return decorator
        task_decorator = MockKBench.task(name="s6e8_automated_feature_discovery")


# %%
def extract_code_block(response_text: str) -> str:
    """
    Extracts executable Python code block containing 'def add_new_features'.
    """
    pattern = r"```(?:python)?\s*(def\s+add_new_features[\s\S]*?)```"
    match = re.search(pattern, response_text, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Fallback to direct substring search
    if "def add_new_features" in response_text:
        idx = response_text.index("def add_new_features")
        return response_text[idx:].split("```")[0].strip()

    raise ValueError("No valid 'def add_new_features' Python function found in LLM response.")


# %%
def evaluate_candidate_features(code_str: str, proxy_sample: int = 15000, n_splits: int = 3) -> float:
    """
    Evaluates proposed mathematical features inside an isolated namespace
    against 3-Fold Stratified CV with Gauss-Rank Stacking.
    """
    namespace = {
        "np": np,
        "pd": pd,
        "scipy": scipy
    }
    # Execute code candidate in isolated namespace
    exec(code_str, namespace)
    if "add_new_features" not in namespace:
        raise AttributeError("Function 'add_new_features' not found in executed code.")

    add_new_features_fn = namespace["add_new_features"]

    train_path = resolve_data_path("train.csv")
    df = pd.read_csv(train_path)
    if proxy_sample and len(df) > proxy_sample:
        df = df.sample(n=proxy_sample, random_state=42).reset_index(drop=True)

    target_col = "addicted_label"
    y = df[target_col]

    # 1. Base Feature Engineering
    X_base = preprocess_and_engineer(df).drop(columns=["id", target_col], errors="ignore")

    # 2. Inject Candidate Features
    X_candidate = add_new_features_fn(X_base.copy()).drop(columns=["id", target_col], errors="ignore")

    # Sanity checks for Shape
    if X_candidate.shape[1] < X_base.shape[1]:
        raise ValueError("Candidate function illegally dropped columns.")

    # 3. Fast Stratified Cross-Validation
    solver = CompetitionSolver(
        n_splits=n_splits,
        random_state=42,
        use_neural_net=True,
        n_estimators=100
    )
    oof_preds_matrix, _ = solver.cross_validate(X_candidate, y)

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


# %%
@task_decorator
def run_feature_discovery_task(llm=None) -> float:
    """
    Closed-loop Automated Feature Engineering (CAAFE / LLM-FE) powered by
    Kaggle Benchmarks Frontier Models (Claude Opus 4.8 / GPT-5.6 Sol).
    """
    prompt = """You are an elite competitive machine learning mathematician optimizing Kaggle S6E8 (Smartphone Addiction Prediction).
Objective: Write a pure Python function 'def add_new_features(df: pd.DataFrame) -> pd.DataFrame' that calculates exactly 2 or 3 high-signal non-linear interaction features.

Dataset & Domain Context:
1. Budget Identity: other_screen = daily_screen_time_hours - (social_media_hours + gaming_hours + work_study_hours)
2. Day Identity: unaccounted_hours = 24 - (daily_screen_time_hours + work_study_hours + sleep_hours)
3. Sleep & Compulsion: sleep_deficit = max(0, 8.0 - sleep_hours), sleep_app_opens_ratio = app_opens_per_day / (sleep_hours + 0.1)
4. Non-Linear Dynamics: Screen time has a sharp risk transition at 5-6 hours.
5. Strict Rule: Never drop existing columns. Handle division by zero with np.where or +1e-5. Do NOT impute NaNs (GBDTs handle missing values natively).

Return ONLY executable Python code starting with ```python and ending with ```."""

    if llm is None:
        print("[STUDIO ENGINE] Engaging Gemini 3.1 Pro (Google AI Studio) as Kaggle Benchmark Evaluator...")
        try:
            from src.studio_engine import get_studio_engine
            engine = get_studio_engine()
            response = engine.generate_text(prompt=prompt, temperature=0.2)
            candidate_code = extract_code_block(response)
        except Exception as e:
            print(f"[WARN] Fallback to simulated candidate: {e}")
            candidate_code = """
def add_new_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['notif_per_open'] = df['notifications_per_day'] / (df['app_opens_per_day'] + 1.0)
    df['addiction_velocity'] = (df['social_media_hours'] + df['gaming_hours']) * df['sleep_deficit']
    return df
"""
    else:
        print("[LLM INVOCATION] Querying Kaggle Model Proxy Frontier Model...")
        response = llm.prompt(prompt)
        candidate_code = extract_code_block(response)

    print("--- Proposing Candidate Code ---")
    print(candidate_code)
    print("--------------------------------")

    try:
        score = evaluate_candidate_features(candidate_code, proxy_sample=10000, n_splits=3)
        print(f"✅ Candidate Evaluation Successful! OOF ROC-AUC: {score:.5f}")
        return float(score)
    except Exception as e:
        print(f"❌ Candidate Execution Failed: {e}")
        traceback.print_exc()
        return 0.0


# %%
if __name__ == "__main__":
    score = run_feature_discovery_task()
    print(f"\nFinal Benchmark Task Output Score: {score}")
