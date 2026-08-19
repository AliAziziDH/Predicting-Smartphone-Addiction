"""
Autonomous Research & Big-Leap Optimization Engine.
Orchestrates:
1. Residual Diagnosis (Visualization & Hard-Sample Profiling)
2. Sequential Hypothesis Reasoning (Google AI Studio - Gemini 3.1 Pro)
3. Evolutionary Feature Discovery & Gating (llm_feature_explorer)
4. Automated Notebook Compilation for Kaggle (compile_notebook)
"""

import os
import sys
import time
from typing import Dict, Any, List

try:
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if ROOT_DIR not in sys.path:
        sys.path.insert(0, ROOT_DIR)
except NameError:
    ROOT_DIR = os.getcwd()

from src.studio_engine import get_studio_engine
from src.llm_feature_explorer import run_evolutionary_search
from src.compile_notebook import compile_notebook


def run_automated_research_cycle(n_generations: int = 1, proxy_sample: int = 10000) -> Dict[str, Any]:
    print("=" * 75)
    print("🚀 FULL AUTOMATED AI RESEARCH & FEATURE OPTIMIZATION CYCLE")
    print("   [Gemini 3.1 Pro + Sequential Reasoning + Residual Mining + Auto Notebook]")
    print("=" * 75)

    start_time = time.time()
    engine = get_studio_engine()

    # Step 1: Sequential Evolutionary Search with Gemini 3.1 Pro
    print("\n[Phase 1/3] 🧠 Executing Sequential Feature Exploration on AI Studio...")
    search_results = run_evolutionary_search(n_generations=n_generations, proxy_sample=proxy_sample)

    baseline_auc = search_results["baseline_auc"]
    best_auc = search_results["best_auc"]
    winning_features = search_results["winning_features"]

    # Step 2: Auto-Compile Kaggle Notebook if new features won
    print("\n[Phase 2/3] 📓 Updating and Compiling Production Kaggle Notebook...")
    notebook_path = os.path.join(ROOT_DIR, "predicting-smartphone-addiction-elite.ipynb")
    compile_notebook(notebook_path)
    print(f"✅ Production Notebook compiled: {notebook_path}")

    # Step 3: Telemetry & Summary
    elapsed = time.time() - start_time
    print("\n" + "=" * 75)
    print("🏁 AUTOMATED RESEARCH CYCLE COMPLETED")
    print("=" * 75)
    print(f"• Baseline AUC:       {baseline_auc:.5f}")
    print(f"• Best Candidate AUC: {best_auc:.5f} (Δ: {best_auc - baseline_auc:+.5f})")
    print(f"• Promoted Features:  {len(winning_features)}")
    print(f"• Total Execution:    {elapsed:.1f}s")
    print(f"\n{engine.get_usage_summary()}")
    print("=" * 75)

    return {
        "baseline_auc": baseline_auc,
        "best_auc": best_auc,
        "promoted_features": winning_features,
        "elapsed_seconds": elapsed
    }


if __name__ == "__main__":
    run_automated_research_cycle(n_generations=1, proxy_sample=5000)
