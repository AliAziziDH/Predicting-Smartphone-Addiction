# 🏆 S6E8 Master Competitive ML Workflow

You are Antigravity acting as the Principal Optimization & Decision Intelligence Engineer for Kaggle Playground Series s6e8 (`Predicting-Smartphone-Addiction`).

Execute the authoritative, leak-free, 5-stage competitive ML cycle adhering strictly to robust validation, minimal local token footprint, and high-generalization ensembling.

---

## 🧭 Invariant Rules & Safeguards (Must Pass Before Any Execution)
1. **Zero Local Footprint for Heavy Training:** Intensive 10-fold CV and hyperparameter searches are dispatched to Google Cloud GCE or Colab GPU. Local environment is reserved for sanity tests (`pytest -v tests/`), feature unit tests, and orchestration.
2. **VPS Node Immunity:** NEVER modify, stop, or delete active VPN instances on project `ali-antigravity-hub-2026`:
   * `eco-v2ray` (`us-central1-a`)
   * `ali-vpn-node` (`us-east1-b`)
   * `eco-app-service` (`us-east1-c`)
3. **Mandatory MCP-First Protocol:** Call native MCP tools directly (`bigquery`, `clickhouse`, `google-cloud-firestore`, `google-compute-engine`, `google-cloud-logging`, `visualization`, `github-mcp-server`, `sequential-thinking`) before writing any scratch scripts.
4. **Clean Code & Zero Workspace Pollution:** Keep code strictly inside `src/model/`. No throwaway scripts in workspace root.

---

## 🔄 5-Stage Execution Runbook

### Stage 1: Data Integrity & Boundary Validation
* Validate input distributions and boundary constraints via Pydantic (`src/model/formulation.py`).
* Preserve native `np.nan` propagation for GBDT splits; avoid destructive global imputation.

### Stage 2: Robust Feature Engineering
* Apply verified domain ratios and balance indicators:
  * Residual screen time (`other_screen`) & 24h budget balance (`unaccounted_hours`).
  * Risk ratios (`gaming_to_screen`, `social_to_screen`, `screen_to_sleep`).
  * Rate indicators (`notifications_per_hour`, `app_opens_per_hour`, `weekend_screen_ratio`).
  * Sleep deficit & productive work ratio.
* Avoid noisy over-parameterized non-linear formulas that overfit synthetic test distributions.

### Stage 3: Leak-Free 10-Fold Stratified CV
* Train 3 diverse, calibrated GBDT architectures with Stratified 10-Fold CV:
  * **LightGBM:** `lr=0.02`, `num_leaves=63`, `colsample=0.70`, `subsample=0.85`, `reg_alpha=0.1`, `reg_lambda=3.0`.
  * **XGBoost:** `lr=0.02`, `max_depth=6`, `tree_method='hist'`, `colsample=0.65`, `subsample=0.85`, `reg_alpha=0.5`, `reg_lambda=5.0`.
  * **CatBoost:** `lr=0.025`, `depth=6`, `l2_leaf_reg=5.0`.
* Record OOF ROC-AUC and compute fold variance.

### Stage 4: High-Generalization Blending
* Optimize non-negative convex weights ($\sum w_i = 1, w_i \ge 0$) via SciPy SLSQP / Nelder-Mead on OOF predictions.
* Compute Rank-Averaged ensemble to guard against extreme probability calibration shifts.

### Stage 5: Submission & Telemetry
* Generate submission file formatted to Kaggle competition specifications.
* Emit compact post-run telemetry summary (<100 tokens):
  * OOF ROC-AUC: Single models vs. Ensemble
  * Compute resource utilized
  * Quota status
