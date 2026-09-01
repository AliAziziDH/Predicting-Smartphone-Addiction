# 📱 Predicting Smartphone Addiction — Production ML & Ensembling Architecture

[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue.svg?style=flat&logo=python)](https://www.python.org/)
[![Pytest Passed](https://img.shields.io/badge/Tests-32%20Passed%20%7C%20100%25-brightgreen.svg?style=flat&logo=pytest)](https://docs.pytest.org/)
[![Pydantic v2](https://img.shields.io/badge/Validation-Pydantic%20v2-red.svg?style=flat&logo=pydantic)](https://docs.pydantic.dev/)
[![Ensemble](https://img.shields.io/badge/Architecture-4--Way%20GBDT%20%2B%20Deep%20Tabular%20MLP-purple.svg?style=flat)]()
[![Optimization](https://img.shields.io/badge/Stacking-SLSQP%20%2B%20Gauss--Rank-orange.svg?style=flat)]()

An end-to-end, production-grade Decision Intelligence and Machine Learning pipeline developed for **Kaggle Playground Series s6e8: Predicting Smartphone Addiction**.

This repository is designed around strict **Separation of Concerns (SoC)**, headless mathematical formulations, leak-free feature pipelines, heterogeneous multi-model ensembling, and deterministic verification.

---

## 🏛️ System Architecture

The architecture enforces strict decoupling between data contracts, mathematical feature engineering, model solvers, and testing suites:

```
Predicting-Smartphone-Addiction/
├── src/
│   ├── model/
│   │   ├── formulation.py       # Pydantic data contracts & leak-free feature engineering
│   │   ├── neural_tabular.py    # Deep Tabular PyTorch MLP & Factorization Machine
│   │   ├── solver.py            # GBDT models, Discrete Target Encoding, Stacking & Drift Screen
│   │   └── tuner.py             # Leak-free Optuna Bayesian hyperparameter search
│   ├── fast_production_runner.py# High-throughput production training & SLSQP blending
│   ├── compile_notebook.py      # Standalone Kaggle submission notebook compiler
│   ├── kaggle_runner.py         # Automated Kaggle Cloud GPU dispatcher & auto-submitter
│   ├── train.py                 # Full 10-Fold Stratified CV training pipeline
│   └── predict.py               # Batch inference & Kolmogorov-Smirnov drift screen
├── tests/
│   ├── test_formulation.py      # Pydantic boundary checks & feature zero-division guards
│   ├── test_blending.py         # Rank stacker & SLSQP convex blending verification
│   ├── test_tuner.py            # Optuna tuner & unseen categories resilience
│   └── test_studio_engine.py    # API mocking & token tracking verification
├── pyproject.toml               # Modern Python project configuration
└── requirements.txt             # Locked production dependencies
```

---

## 🔬 Core Engineering Innovations & Visual Insights

### 1. Feature Signal Emergence & Domain Ratios
Tree-based gradient boosters benefit substantially from native branch routing on missing values. Destructive global imputations (mean, median) are strictly avoided in favor of mathematically grounded domain balances:
- **24-Hour Life Budget Balance:** Residual unallocated time ($\text{Budget} = 24 - (\text{Screen} + \text{Work} + \text{Sleep})$).
- **Residual Screen Time:** Screen time not accounted for by social, gaming, or work apps.
- **Compulsive Pull Ratio:** Interaction checking intensity ($\text{App Opens} / (\text{Notifications} + 1)$).
- **Work Shield Factor:** Protective mitigation of productive work against addiction probability ($\frac{\text{Work}}{\text{Screen} + \epsilon} \cdot \exp(-\frac{\text{Leisure}}{2.5})$).

<p align="center">
  <img src="assets/figure1_feature_distributions.png" alt="Feature Signal Emergence" width="95%"/>
</p>

### 2. Discrete Categorical Target Encoding
Applies internal 5-fold Out-of-Fold (OOF) target and frequency encoding strictly to discrete categorical pairs (e.g., `gender × stress_level`) with **Laplace smoothing ($\text{smooth}=20.0$)** to eliminate target leakage and guard against low-cardinality overfitting.

### 3. Heterogeneous 4-Way Modeling & Residual Diversity
Combines diverse predictive models across distinct hypothesis spaces to maximize ensemble diversity:
- **LightGBM:** Fast histogram-based gradient boosting with path smoothing and regularization.
- **XGBoost:** Exact depth-constrained trees with heavy $L_1/L_2$ regularization.
- **CatBoost:** Oblivious decision trees with symmetric structure and Bernoulli subsampling.
- **Deep Tabular Neural Network:** Continuous representation layer capturing non-tree manifolds.

<p align="center">
  <img src="assets/figure2_ensemble_diversity.png" alt="Heterogeneous Ensemble Diversity Heatmap" width="60%"/>
</p>

### 4. SLSQP Convex Blending & Gauss-Rank Meta-Stacking
Heterogeneous model predictions are transformed via **Empirical Cumulative Distribution Function (ECDF)** percentiles to eliminate calibration discrepancies across model families:
$$\text{Rank}(p_i) = \frac{\text{rank}(p_i) - 0.5}{N}, \quad z_i = \Phi^{-1}(\text{Rank}(p_i))$$
Optimal ensembling weights are then solved via **Sequential Least Squares Programming (SLSQP)** under non-negativity and sum-to-one constraints:
$$\min_{\mathbf{w}} -\text{ROC-AUC}\left(\sum_{m} w_m \cdot \text{Rank}(p^{(m)}), y\right) \quad \text{s.t.} \quad \sum w_i = 1, \quad w_i \ge 0$$

<p align="center">
  <img src="assets/figure3_rank_transformation_ks.png" alt="Gauss-Rank Transformation" width="85%"/>
</p>

### 5. Kolmogorov-Smirnov (KS) Distribution Drift Guardrail
During inference, a two-sample Kolmogorov-Smirnov test is computed between Out-of-Fold predictions and test set predictions ($D_{\text{crit}} < 0.03$) to detect and prevent shake-up before final submission generation.

---

## ⚡ Quickstart & Reproducibility

### 1. Installation
```bash
# Clone the repository
git clone https://github.com/AliAziziDH/Predicting-Smartphone-Addiction.git
cd Predicting-Smartphone-Addiction

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Verification Test Suite
All data contracts, mathematical boundary guards, and ensembling algorithms are validated via Pytest:
```bash
pytest -v
# 32 passed, 100% test coverage
```

### 3. Local Training & Inference
```bash
# Execute fast 5-fold / 10-fold Stratified CV with SLSQP rank blending
python src/fast_production_runner.py --folds 5 --no-gcs

# Or run full headless training and inference
python src/train.py
python src/predict.py
```

### 4. Compile Standalone Kaggle Notebook
To compile the entire modular codebase into a single, self-contained Kaggle submission notebook:
```bash
python src/compile_notebook.py
# Generates predicting-smartphone-addiction-elite.ipynb
```

---

## 📈 The Optimization Journey: From Baseline (0.949) to Peak Meta-Blend (0.96943)

Competitive machine learning is rarely about a single model; it is an iterative optimization trajectory. Here is how the architecture evolved over 12 development waves:

```
[Wave 1-5: Naive GBDT + Imputation] ──► 0.94979
             │
             ▼
[Wave 6-7: 24h Life Budget & Domain Ratios] ──► 0.96434 (+0.0145)
             │
             ▼
[Wave 8: Out-of-Fold Discrete Target Encoding] ──► 0.96469
             │
             ▼
[Wave 9: 54 Scaled Features + Nelder-Mead Rank Stack] ──► 0.96919
             │
             ▼
[Wave 9 + 10: Multi-Wave Rank Blending (Peak Formulation)] ──► 0.96943 Public / 0.96920 Private
```

### 🗺️ Iterative Progression Table

| Development Wave / Strategy | Key Architectural Innovation | Public ROC-AUC | Private ROC-AUC | Key Lesson |
| :--- | :--- | :---: | :---: | :--- |
| **Initial GBDT Baselines** | Raw features, global mean imputation | `0.94979` | `0.95015` | Destructive imputation damages tree branch decisions. |
| **Wave 6: Domain Formulation** | 24h life budget, residual screen time, native NaNs | `0.96434` | `0.96430` | Domain physics (time budget constraints) produce massive signal gain. |
| **Wave 7: Discrete Target Encoding** | 5-Fold internal OOF encoding with Laplace smoothing | `0.96440` | `0.96436` | Encoding categorical pairs (`gender × stress`) captures non-linear interactions. |
| **Wave 8: Regularized 10-Fold CV** | Deep tree regularization + early stopping | `0.96463` | `0.96469` | High-capacity trees require aggressive path smoothing ($L_1/L_2$). |
| **Wave 9: Discrete Lattice & Stacking** | 54 features + Nelder-Mead Gauss-Rank stacker | `0.96941` | `0.96919` | Gauss-Rank percentiles eliminate cross-model calibration shifts. |
| **Wave 10: StudioEngine Discovery** | 63 features + continuous Deep Tabular MLP | `0.96941` | `0.96917` | Neural networks provide essential non-tree continuous representation. |
| **🏆 Wave 9 + 10 Meta-Blend** | **Multi-Wave SLSQP Bounded Rank Ensemble** | **`0.96943`** | **`0.96920`** | **Optimal ensembling of diverse feature sets yields peak generalization.** |

---

### 💡 Key Takeaways & Post-Mortem Insights

1. **Feature Engineering Over Hyperparameter Tuning:**  
   The single largest jump ($+0.015$ AUC) came from formulating **closed time-budget constraints** and **interaction intensity ratios**, far outperforming raw grid search on default features.
2. **Preserve Native Missingness:**  
   Allowing gradient boosting split finders to route `np.nan` values natively preserved critical missingness information that global imputation destroyed.
3. **Rank-Space Ensembling Guards Against Shift:**  
   Averaging raw probabilities often suffers when models have different calibration slopes. Converting predictions to empirical percentiles ($\text{Rank}(p)$) before SLSQP optimization provided total resilience against distribution shift.

---

## 👨‍💻 Principal Architect

**Ali Azizi**  
*Decision Intelligence & Machine Learning Optimization Engineer*  
Sharif University of Technology (MSc Optimization)

---

## 📄 License
This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
