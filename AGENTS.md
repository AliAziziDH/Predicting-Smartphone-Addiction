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
   - Solver invocation, cross-validation logic, stacking/ensembling, and Out-of-Fold (OOF) predictions safely avoiding target leakage belong strictly in `src/model/solver.py`.
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

## 🧪 Testing and Verification Suite
Jules is an edit-test-repair agent. It cannot verify code changes without a
deterministic pass/fail check.
- **Test Runner:** Always use `pytest` as the verification engine.
- **Execution Command:** Run `pytest -v` from the repository root to verify edits.
- **Standard Assertions for Machine Learning:**
  - `test_pydantic_validation`: Validates input boundary constraints.
  - `test_engineered_features_math`: Verifies that engineered ratio features do not divide by zero and yield correct dimensions (16 total columns).
  - `test_leak_free_cv`: Asserts that validation folds remain completely untouched during fit cycles.

---

## 🛡️ Repository Etiquette & Pull Request Safety
- **Draft PRs:** Always open Pull Requests as Drafts by default. Merges require
  explicit human-in-the-loop review.
- **Mentions Only:** Jules only responds to review comments that explicitly
  mention `@jules` to preserve token quotas.
