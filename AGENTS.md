# 📄 AGENTS.md: Repository Capabilities, Architecture, and Agent Guidelines

This repository is optimized for agentic development (Google Jules, Claude Code).
All agents operating on this repository MUST strictly adhere to the standards,
architecture, and execution rules defined in this document.

---

## 🎯 Project Overview & Context
- **Project Name:** playground-series-s6e8
- **Target Domain:** Kaggle / Decision Intelligence / Machine Learning
- **Principal Architect:** Ali Azizi (Decision Intelligence Engineer, Sharif University MSc Optimization)
- **Primary Tech Stack:** Python 3.11+, LightGBM, XGBoost, CatBoost, Scikit-Learn, Pydantic, Pytest

---

## 🏛️ System Architecture & Separation of Concerns (SoC)
To prevent agent drift and maintain modular, clean code, this repository strictly
enforces a separation of concerns, strictly decoupling the predictive layer from any future prescriptive or Streamlit UI layers (Matt Pocock SoC principles).

1. **Analytical Core (`src/model/`):**
   - Must contain pure mathematical formulations and data-science pipelines.
   - Pydantic models for strict boundary validation and feature engineering MUST be in `src/model/formulation.py`.
   - GBDT solvers (LightGBM, XGBoost, CatBoost), Deep Tabular PyTorch Neural Networks (`src/model/neural_tabular.py`), Vertex AI AutoML integration (`src/model/vertex_automl.py`), stacking/ensembling, and Out-of-Fold (OOF) predictions belong strictly in `src/model/solver.py`.
   - The analytical core must remain 100% headless (no Streamlit, no Matplotlib UI).

2. **Testing Core (`tests/`):**
   - All sanity checks, math verification, and testing of Pydantic components belong in `tests/`.

---

## 🔌 Stateless MCP 2.0 & Token Economy Directives
- **Model Context Protocol (Spec 2026-07-28):** All tools and agent interactions
  use stateless execution.
- **Tools Tax Elimination (Tool Attention):** Do not load raw JSON Schemas of all
  tools. Utilize compact tool summaries (<60 tokens) for initial discovery.
- **Code Execution on MCP:** Write compact, self-contained Python scripts that
  execute local tool chains in a sandboxed executor, reducing input token
  overhead by 78.5% and costs by 70%.
- **Explicit Handles:** Return explicit resource identifiers (e.g., `dataset_id`,
  `model_id`) in tool outputs and require the agent to pass them back in later
  calls, keeping the system serverless-friendly.

---

## ⚡ Compute & Cloud Execution Policy (Colab / GCP / Kaggle Dual-Engine)
- **Zero Local Footprint Policy:** Heavy 10-fold cross-validation, hyperparameter searches (Optuna), and deep model training MUST NOT be run on the local machine CPU or waste Kaggle GPU quotas.
- **External Training & Quota Preservation Engine (Colab / GCP Sandbox):** Extensive training workflows, LLM-based automated feature extraction, and exploratory neural fits are dispatched to Google Colab GPU (`google-colab-cli`) or Google Cloud Compute Engine (`google-compute-engine` on project `ali-antigravity-hub-2026`). This strictly preserves weekly Kaggle GPU quotas.
- **Google AI Studio Intelligence Core (`src/studio_engine.py`):** All generative feature discovery, residual diagnostics, and mathematical formulations MUST strictly leverage `gemini-3.1-pro-preview` in Google AI Studio via `StudioEngine`.
- **Kaggle Dual-T4 Engine:** Reserved exclusively for final verified submissions, production inference pipelines (<1 min runtime), and official Kaggle Benchmark evaluations.
- **Local Role:** Strictly reserved for instant sanity tests (`pytest -v`), AST linting, and dispatch orchestration.

---

## 🚀 Official Google Colab CLI & Colab MCP Integration Guide
This repository utilizes official Google tooling for seamless remote GPU orchestration:
1. **Google Colab CLI (`google-colab-cli`):**
   - **GPU Provisioning:** `colab new --gpu T4 -s s6e8` (Allocates Nvidia Tesla T4 GPU cloud VM).
   - **Environment Bootstrap:** `colab install -s s6e8 lightgbm xgboost catboost pydantic optuna scipy scikit-learn kaggle`
   - **Fast Asset Transfer:** `colab upload -s s6e8 <local_file> <remote_path>` / `colab download -s s6e8 <remote_file>`
   - **Headless Execution:** `colab exec -s s6e8 -f <script.py>`
   - **Session Lifecycle:** `colab status -s s6e8`, `colab stop -s s6e8`
   - **Patches Applied:** `jupyter_kernel_client.KernelClient = JupyterKernelClient`, `REQUEST_TIMEOUT = 3600` for long training cycles.
2. **Google Colab MCP Server (`googlecolab/colab-mcp`):**
   - Official MCP protocol bridging Antigravity and AI agents to active Colab sessions (`list_sessions`, `fetch_transcript`, `search_logs`, `summarize_session`).
   - Configured via `"colab-mcp": {"command": "uvx", "args": ["git+https://github.com/googlecolab/colab-mcp"]}`.
3. **Execution Invariant:**
   - Always verify active GPU session with `colab status -s <name>`.
   - Never allow unmonitored local CPU loops when Colab GPU is available.

---

## 📊 6-Tool Cloud Infrastructure & Post-Run Telemetry Protocol
After every major training, optimization, or feature discovery run, the agent MUST adhere to the following 6-tool cloud pipeline and emit a structured, ultra-compact telemetry report:
1. **Compute (`google-compute-engine` / `cloudrun`):** Runs headless workloads on cloud instances without blocking the local workspace.
2. **Logging (`google-cloud-logging`):** Redirects verbose epoch/fold logs to cloud logs or local log files, preventing chat context pollution (<500 tokens/turn).
3. **Monitoring & Quotas (`google-cloud-monitoring` + `google-cloud-quotas`):** Audits memory, VRAM footprint, and remaining GCP/Kaggle quotas.
4. **Post-Run Telemetry Output Template (Mandatory after each run):**
   ```text
   📊 [Post-Run Telemetry & Quota Report]
   • Cloud Resource: GCE / ali-antigravity-hub-2026 (Status: Healthy / Inactive)
   • Resource Footprint: RAM: <X> MB | VRAM: <Y> GB | Execution Time: <Z>s
   • Studio Engine: Gemini 3.1 Pro (Calls: <N> | Tokens: <T>)
   • Kaggle GPU Quota: Preserved (Used: 0s / 30h)
   ```

---

## 🗄️ 4-Tier Data & Analytics Architecture (SoC Gating)
To maximize mathematical precision and maintain an evolving intelligence pipeline, the agent MUST leverage the 4 specialized data tiers:
1. **BigQuery (`bigquery`):** Heavy batch feature engineering, large-scale statistical aggregations, Target Encoding distributions, and dataset validation.
2. **ClickHouse (`clickhouse`):** Fast in-memory/columnar screening of feature correlation matrices, quantile distributions, and real-time interaction benchmarks.
3. **Cloud Firestore (`google-cloud-firestore`):** Evolving Feature Store & Prompt Vault — stores all feature candidates, Python AST formulations, and mathematical rationales synthesized by Gemini 3.1 Pro as structured JSON documents.
4. **Cloud SQL (`cloud-sql`):** Relational Experiment Tracking Ledger — records trial hyperparameters (Optuna), fold-by-fold OOF AUC scores, execution timestamps, and Kaggle submission IDs.

---

## 🧪 Testing and Verification Suite
Jules is an edit-test-repair agent. It cannot verify code changes without a
deterministic pass/fail check.
- **Test Runner:** Always use `pytest` as the verification engine.
- **Execution Command:** Run `pytest -v` from the repository root to verify edits.
- **Standard Assertions for Machine Learning:**
  - `test_pydantic_validation`: Validates input boundary constraints.
  - `test_engineered_features_math`: Verifies that engineered ratio features do not divide by zero and yield correct dimensions (16 total columns).
  - `test_leak_free_cv`: Asserts that validation folds remain completely untouched during fit cycles.
  - `test_studio_engine`: Asserts Google AI Studio integration passes with proper token tracking.

---

## 🛡️ Repository Etiquette & Pull Request Safety
- **Draft PRs:** Always open Pull Requests as Drafts by default. Merges require
  explicit human-in-the-loop review.
- **Mentions Only:** Jules only responds to review comments that explicitly
  mention `@jules` to preserve token quotas.

---

## 🧬 Multi-Agent Evolutionary Feature Discovery & Wave 2 Heuristics
- **Automated Feature Explorer (`src/llm_feature_explorer.py`):** Powered by `StudioEngine` (Gemini 3.1 Pro) and operates on a Two-Stage Promotion Gate:
  1. **Stage 1 (Fast Sandbox Screening):** 15k proxy samples, 3-Fold Stratified CV, and Gauss-Rank Stacking in <5s.
  2. **Stage 2 (Full Promotion):** Complete 10-Fold OOF CV before permanent commit into `src/model/formulation.py`.
- **Wave 2 Mathematical Formulations:**
  1. **Joint Profile Frequency (`joint_profile_freq`):** Scale-invariant `normalize=True` density for `gender + '_' + stress_level + '_' + academic_work_impact`.
  2. **Nonlinear Risk Boundary Distance:** `np.abs(daily_screen_time_hours - 5.5) * (social + gaming) / (24 - sleep_hours)`.
  3. **Group-Normalized Cohort Residuals:** `social_media_hours - groupby(['age', 'gender'])['social_media_hours'].transform('mean')`.
- **Automated Master Cycle (`src/auto_research_cycle.py`):** Fully integrates sequential reasoning, Google AI Studio feature discovery, residual error diagnostics, and automated Kaggle notebook compilation into a one-command pipeline.
- **Native MCP Directives:** Utilize `google-compute-engine` and `bigquery` for cloud compute, `notebooks` for automated Jupyter synchronization, `sequential-thinking` for deep mathematical synthesis, and `visualization` for residual error diagnostics.

---

## 🛠️ DevOps, Web, Security & Autonomous Tool Synergies
To ensure complete resilience, zero information loss, and automated synchronization across all 15 tools:
1. **GitHub Version Control (`github-mcp-server`):** Every promoted feature iteration or verified Kaggle submission is automatically staged, committed, and synced to GitHub (`AliAziziDH/Predicting-Smartphone-Addiction`).
2. **Kaggle Live Scraping & Web Inspection (`chrome-devtools-mcp` + `chrome-devtools-plugin`):** Live monitoring of competition leaderboard shifts, public top notebook trends, and submission status audits directly via Chrome DevTools.
3. **Security & Package Integrity (`sonatype-guide`):** Pre-execution verification of package compatibility and dependency health before introducing new ML libraries.
4. **Autonomous AI Telemetry Advisor (`src/ai_telemetry_monitor.py`):** Real-time monitoring of fold variance, stacker coefficients, and automated post-run telemetry reporting.


