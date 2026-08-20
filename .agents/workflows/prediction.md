# 🏆 S6E8 Smartphone Addiction Prediction Master Workflow

You are Antigravity acting as the Principal Optimization & Decision Intelligence Engineer for Kaggle Playground Series s6e8 (`Predicting-Smartphone-Addiction`).

Execute the authoritative, 7-stage master engineering cycle adhering strictly to repository architecture, 4-tier data gating, zero local footprint, and Top-100 ensembling protocols:

---

## 🧭 Invariant Rules & Safeguards (Must Pass Before Any Execution)
1. **Zero Local Footprint:** NEVER run 10-fold CV, Optuna trials, or deep tree training on the local Mac CPU. All heavy computation is dispatched to Google Cloud GCE (`google-compute-engine`) or Google Colab GPU (`google-colab-cli`). Local environment is strictly for instant AST verification (`pytest -v tests/`).
2. **VPS Node Immunity:** NEVER modify, stop, or delete the 3 active VPN instances on project `ali-antigravity-hub-2026`:
   * `eco-v2ray` (`us-central1-a`)
   * `ali-vpn-node` (`us-east1-b`)
   * `eco-app-service` (`us-east1-c`)
3. **Mandatory MCP-First Protocol:** Call native MCP tools directly (`bigquery`, `clickhouse`, `google-cloud-firestore`, `google-compute-engine`, `google-cloud-logging`, `visualization`, `github-mcp-server`, `sequential-thinking`) before writing any scratch scripts.
4. **Clean Code & Zero Workspace Pollution:** Never create temporary scratch scripts in the workspace root. Keep code strictly inside `src/model/`.

---

## 🔄 7-Stage Execution Runbook

### Stage 1: 4-Tier Data Analytics & Matrix Screening
* Audit feature correlation distributions via in-memory columnar screening in `clickhouse`.
* Extract group-normalized cohort statistics and Target Encoding distributions across `[age, gender]` using `bigquery`.
* Store and retrieve candidate feature formulations from Cloud Firestore (`google-cloud-firestore`) Feature Store.

### Stage 2: Generative Discovery (StudioEngine & Gemini 3.1 Pro)
* Analyze residual errors, hardest-5% CTGAN transition boundary samples (`screen_time = 5.78`), and multi-missing cohorts.
* Formulate continuous interactions (`st * sm`, `st / (24 - sl)`, `(st - 5.5) * (sm + gm)`) and 2-way discrete categoricals (`gender x stress`, `gender x impact`, `stress x impact`).
* Enforce **Discrete 2-Way Target Encoding Rule:** Continuous 2-way TE is prohibited to prevent 15,000+ sparse levels. Discrete 2-way TE must use `smooth=20.0`.

### Stage 3: Two-Stage Promotion Gate
* **Gate 1 (Fast Proxy Screening):** 15k proxy samples, 3-Fold Stratified CV, Gauss-Rank meta-stacking in <3s. Reject candidate if $\Delta \text{AUC} < +0.0003$ or Kolmogorov-Smirnov (KS) test indicates train/test drift ($p < 0.05$).
* **Gate 2 (Master Benchmark Gate):** 30k proxy samples across LightGBM, XGBoost, and CatBoost.

### Stage 4: Code Integration & Local Verification
* Update `src/model/formulation.py` with verified feature transformations.
* Update `src/model/solver.py` with vectorized NumPy array Target Encoders (<0.2s runtime).
* Verify 100% test pass rate via `pytest -v tests/`.
* Commit and push clean diffs to GitHub (`AliAziziDH/Predicting-Smartphone-Addiction`).

### Stage 5: Remote Cloud Execution & Checkpointing
* Provision an `e2-standard-8` (or Nvidia GPU) worker in `us-central1-c`.
* Train full 10-fold CV with calibrated production hyperparameters:
  * **LightGBM:** `lr=0.025`, `n_estimators=2500`, `num_leaves=63`, `colsample=0.70`, `subsample=0.85`, `reg_alpha=0.1`, `reg_lambda=5.0`, `early_stopping=80`.
  * **XGBoost:** `lr=0.025`, `n_estimators=2500`, `max_depth=6`, `tree_method='hist'`, `colsample=0.65`, `subsample=0.85`, `reg_alpha=0.5`, `reg_lambda=8.0`, `early_stopping=80`.
  * **CatBoost:** `lr=0.03`, `iterations=2200`, `depth=6`, `l2_leaf_reg=6.0`, `od_type='Iter'`, `od_wait=80`.
* Persist `fold_k_checkpoint.npz` files (OOF probabilities, test predictions, validation indices) to `gs://ali-s6e8-kaggle-artifacts-2026/checkpoints/`.

### Stage 6: Rank-Gauss Meta-Stacking & Multi-Wave Blending
* Project OOF prediction probabilities to standard normal Gaussian space via Rank-Gauss:
  $$z_m = \Phi^{-1}\left( \frac{\text{Rank}(\hat{y}_m) - 0.5}{N} \right)$$
* Fit non-negative convex stacking weights ($\sum w_m = 1$) using SLSQP / Regularized Ridge.
* Blend current wave OOF predictions with previous wave checkpoints (e.g. Wave 9 + Wave 10) to reduce model variance and unlock the Top-100 threshold (`0.9710+`).

### Stage 7: Automated Kaggle Submission & Telemetry Reporting
* Fetch final submission array directly from GCS bucket.
* Submit via Kaggle API: `kaggle competitions submit -c playground-series-s6e8 -f submission_elite_wave*.csv -m "<description>"`.
* Inspect public leaderboard score and rank shifts.
* Emit mandatory 6-tool telemetry report.
