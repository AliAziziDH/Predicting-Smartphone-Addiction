# 📘 CLOUD_INFRASTRUCTURE_PLAYBOOK.md: Comprehensive Cloud Execution & Optimization Architecture

This document serves as the permanent, authoritative reference manual and operational playbook for the `ali-antigravity-hub-2026` cloud environment in competition `playground-series-s6e8`. It integrates all architectural decisions, financial safeguards, and code blueprints developed in synergy with **Google AI Studio (Gemini 3.1 Pro)**.

---

## 🏛️ System Architecture Overview

```text
                           [ Google AI Studio ]
                        (Gemini 3.1 Pro Architect)
                                    │
                                    ▼
                [ src/llm_feature_explorer.py / StudioEngine ]
                                    │
         ┌──────────────────────────┴──────────────────────────┐
         ▼                                                     ▼
 [ Local Fast Sandbox ]                                [ Cloud Spot VMs ]
 • AST Linting & Validation                            • 10-Fold Stratified CV
 • 15k Proxy Downsampling (<2s)                        • Distributed Bayesian Optuna
 • Fast Rank-AUC Screening                             • Preemption-Proof Warm-Start
         │                                                     │
         └──────────────────────────┬──────────────────────────┘
                                    ▼
                 [ Cloud Storage & PostgreSQL Ledger ]
 • GCS Bucket: `gs://ali-s6e8-kaggle-artifacts-2026/` (OOF & Checkpoints)
 • Cloud SQL: PostgreSQL 18 `ai-studio-598ceb1b` (Optuna Trial Tracking)
 • Optuna Dashboard: Local Real-Time Telemetry GUI (`:8080`)
```

---

## 💰 1. Financial Governance & Budget Safeguards ($280 Balance)

### A. Granular $10-Interval Budget Alerts
* **Budget Bucket Name:** `Kaggle-S6E8-10Dollar-Intervals`
* **Billing Account ID:** `01F483-F84F5E-B7E0A6`
* **Policy:** Triggers an alert at every **$10 increment** of spend (e.g. $10, $20, $30 ... $280).
* **Zero Forced Interruption:** No automatic destructive shutdowns (`poweroff`/`kill`) occur without explicit confirmation from Ali Azizi.

### B. Minimal Disk Footprint Doctrine
* **Boot Disk Sizing:** Cap Spot VM boot disks strictly at **30GB Standard Persistent Disk (PD)**.
* **Role:** Purely OS and temporary fold-level scratch space.
* **Storage Offloading:** 100% of large Parquet datasets, OOF probability arrays, and model weights are written/synced directly to Cloud Storage.

---

## 🛡️ 2. Preemption Recovery & Resilience Architecture

### A. SIGTERM Preemption Watchdog
When GCP reclaims a Spot instance, it issues a 30-second `SIGTERM` notice. The runner captures this signal and flushes emergency checkpoints to GCS before termination:

```python
# src/fast_production_runner.py
import signal
import subprocess
import sys

def handle_sigterm(signum, frame):
    print("\n🚨 [PREEMPTION WATCHDOG] SIGTERM received! Flushing checkpoints to GCS...")
    if gcs_bucket:
        try:
            subprocess.run(["gcloud", "storage", "cp", "-r", checkpoint_dir, gcs_bucket], capture_output=True)
            print(f"✅ Emergency checkpoint successfully synchronized to {gcs_bucket}")
        except Exception as e:
            print(f"⚠️ Failed to sync emergency checkpoint: {e}")
    sys.exit(0)

signal.signal(signal.SIGTERM, handle_sigterm)
```

### B. Optuna Warm-Start Ledger (`src/model/solver.py`)
To prevent lost hyperparameter search iterations during VM preemption:

```python
# src/model/solver.py
def get_or_create_cloud_study(study_name: str = "s6e8_master_study") -> optuna.Study:
    database_url = f"postgresql+psycopg2://{db_user}:{db_pass}@{db_host}:5432/{db_name}"
    storage = RDBStorage(
        url=database_url,
        engine_kwargs={
            "pool_size": 10,
            "max_overflow": 20,
            "pool_pre_ping": True  # Prevents stale connections
        }
    )
    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        storage=storage,
        load_if_exists=True    # 👈 Resumes from last trial if interrupted
    )
    return study
```

---

## ⚡ 3. Cloud Storage FUSE (`gcsfuse`) Operational Guide

### A. High-Throughput Mounting Command
To mount `gs://ali-s6e8-kaggle-artifacts-2026/` locally on Spot VMs with aggressive caching:

```bash
gcsfuse \
  --implicit-dirs \
  --stat-cache-ttl 1h \
  --type-cache-ttl 1h \
  --file-cache-ttl 1h \
  --max-conns-per-host 100 \
  --dir-mode 0777 \
  --file-mode 0777 \
  ali-s6e8-kaggle-artifacts-2026 /mnt/artifacts/
```

### B. Critical Architectural Gotchas
1. ⚠️ **NO SQLite on gcsfuse:** `gcsfuse` does NOT support POSIX file locking. Never place an Optuna SQLite file on `/mnt/artifacts/`. Use Cloud SQL PostgreSQL instead.
2. ⚠️ **Sequential Parquet Writes Only:** Write OOF arrays as batched Parquet files (`pl.DataFrame.write_parquet`) rather than frequent small random writes.

---

## 📊 4. Real-Time Telemetry & Optuna Dashboard

To inspect hyperparameter distributions, convergence curves, and feature importance in real-time from your local machine:

```bash
# 1. Install dashboard
pip install optuna-dashboard

# 2. Launch Local GUI connected to Cloud SQL
optuna-dashboard postgresql+psycopg2://$CLOUD_SQL_USER:$CLOUD_SQL_PASSWORD@$CLOUD_SQL_HOST:5432/$CLOUD_SQL_DB
```
*Access GUI at:* `http://127.0.0.1:8080`

---

## 🎯 5. Phased Execution Roadmap

| Phase | Core Objective | Engine & Hardware | Status |
| :--- | :--- | :--- | :---: |
| **Phase 1: Baseline 10-Fold OOF** | Full 691k training with 41 Wave 8 features & Early Stopping | Colab Tesla T4 GPU / GCE | 🟡 Running (8/10 Folds done) |
| **Phase 2: Distributed Optuna Search** | Multi-worker Bayesian tuning with Cloud SQL Warm-Start | GCE Spot VMs (`e2-highmem-16`) | 🟢 Ready for Launch |
| **Phase 3: Tabular Deep Neural Fits** | Multi-epoch PyTorch embeddings + Nelder-Mead Rank Stacking | GCE Spot GPU (T4/L4) | ⏳ Staged for Phase 3 |

---

## 🗂️ 6. Key File Index

- `src/model/formulation.py`: 41 engineered features, vector-safe calculations, CTGAN deficit metrics.
- `src/model/solver.py`: Core solvers, Nelder-Mead Rank Stacker, `get_or_create_cloud_study`.
- `src/fast_production_runner.py`: 10-Fold production runner with early stopping and SIGTERM watchdog.
- `src/distributed_cloud_tuner.py`: Multi-model Optuna tuning engine with Cloud SQL ledger support.
- `docs/CLOUD_CREDIT_OPTIMIZATION.md`: Master strategy charter.
