# S6E8 Master Competitive Cycle Workflow

Execute the full, autonomous 7-stage machine learning development cycle for Kaggle Playground Series s6e8 (`Predicting-Smartphone-Addiction`):

1. **Pre-Execution Invariants Audit:**
   - Verify zero local heavy computation on Mac CPU.
   - Assert all 3 GCP VPS nodes (`eco-v2ray`, `ali-vpn-node`, `eco-app-service`) remain 100% untouched.
   - Enforce mandatory MCP-First tool routing protocol.

2. **4-Tier Data Exploration & Screening:**
   - Screen candidate interaction formulas in `clickhouse` (in-memory columnar matrix).
   - Compute group-normalized cohort residuals across age/gender in `bigquery`.
   - Audit feature definitions in `google-cloud-firestore` Feature Store.

3. **Generative Feature Formulation (StudioEngine):**
   - Synthesize non-linear domain boundaries and residual error patterns with Gemini 3.1 Pro via `StudioEngine`.
   - Apply Discrete 2-Way Target Encodings strictly to low-cardinality discrete pairs (`gender x stress`, `gender x impact`, `stress x impact`) with `smooth=20.0`.

4. **Two-Stage Promotion Gate:**
   - Fast screening on 15k proxy subset (3-fold CV) in <3s.
   - Reject any feature with $\Delta \text{AUC} < +0.0003$ or Kolmogorov-Smirnov drift ($p < 0.05$).
   - Full 30k proxy benchmark across LightGBM, XGBoost, and CatBoost.

5. **Clean Code Formulation & Verification:**
   - Integrate approved features into `src/model/formulation.py`.
   - Vectorize Target Encodings with NumPy arrays in `src/model/solver.py`.
   - Run local unit tests: `pytest -v tests/` (must pass 100%).
   - Commit and push clean updates to GitHub.

6. **Remote Cloud Training & Checkpointing:**
   - Provision `e2-standard-8` (or Nvidia GPU) instance on Google Cloud in `us-central1-c`.
   - Train 10-fold CV with calibrated parameters (`lr=0.025`, `n_estimators=2500`, `early_stopping=80`, `colsample=0.65-0.70`).
   - Sync checkpoints to `gs://ali-s6e8-kaggle-artifacts-2026/checkpoints/`.

7. **Rank-Gauss Meta-Stacking, Multi-Wave Blending & Kaggle Submission:**
   - Transform OOF probabilities via Rank-Gauss and solve SLSQP convex weights.
   - Blend with previous wave checkpoints (Wave 9 + Wave 10) to minimize model variance.
   - Submit `submission_elite_wave*.csv` to Kaggle API.
   - Emit structured 6-tool telemetry report.
