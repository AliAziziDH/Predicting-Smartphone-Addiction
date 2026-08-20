import os
import json
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional, List
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from scipy.stats import ks_2samp, norm
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
import optuna
from scipy.optimize import minimize

from src.model.formulation import preprocess_and_engineer
from src.model.neural_tabular import DeepTabularClassifier


# Highly calibrated, regularized GBDT configurations for synthetic tabular manifolds
LGBM_PARAMS = {
    "objective": "binary",
    "metric": "auc",
    "boosting_type": "gbdt",
    "n_estimators": 3000,
    "learning_rate": 0.015,
    "num_leaves": 127,
    "max_depth": -1,
    "min_child_samples": 250,
    "subsample": 0.8,
    "colsample_bytree": 0.7,
    "reg_alpha": 0.5,
    "reg_lambda": 2.0,
    "random_state": 42,
    "verbose": -1,
    "n_jobs": -1
}

XGB_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "n_estimators": 3000,
    "learning_rate": 0.015,
    "max_depth": 7,
    "min_child_weight": 15,
    "subsample": 0.8,
    "colsample_bytree": 0.75,
    "reg_alpha": 0.5,
    "reg_lambda": 5.0,
    "random_state": 42,
    "tree_method": "hist",
    "n_jobs": -1
}

CAT_PARAMS = {
    "loss_function": "Logloss",
    "eval_metric": "AUC",
    "iterations": 3000,
    "learning_rate": 0.02,
    "depth": 7,
    "l2_leaf_reg": 5.0,
    "bootstrap_type": "Bernoulli",
    "subsample": 0.8,
    "random_state": 42,
    "verbose": False,
    "thread_count": -1
}

def get_calibrated_model_params():
    """Dynamically routes parameters to CUDA GPU and integrates Optuna tuned parameters if available."""
    lgb_p = LGBM_PARAMS.copy()
    xgb_p = XGB_PARAMS.copy()
    cat_p = CAT_PARAMS.copy()

    # Load tuned parameters from Optuna if present
    tuned_path = os.path.join(os.getcwd(), "models", "best_gbdt_params.json")
    if os.path.exists(tuned_path):
        try:
            with open(tuned_path, "r") as f:
                tuned_data = json.load(f)
            if "lgb_params" in tuned_data:
                lgb_p.update(tuned_data["lgb_params"])
            if "xgb_params" in tuned_data:
                xgb_p.update(tuned_data["xgb_params"])
            print("[TUNER] 🎯 Loaded Optuna-tuned hyperparameters into model solver.", flush=True)
        except Exception as e:
            print(f"[TUNER WARN] Failed to load tuned params: {e}", flush=True)

    try:
        import torch
        has_cuda = torch.cuda.is_available()
    except Exception:
        has_cuda = False

    if has_cuda:
        xgb_p["device"] = "cuda"
        cat_p["task_type"] = "GPU"
        print("[HARDWARE] 🚀 Nvidia CUDA GPU detected! Accelerated hist engine engaged.", flush=True)
    else:
        xgb_p["device"] = "cpu"
        cat_p["task_type"] = "CPU"

    return lgb_p, xgb_p, cat_p


class UniversalLevelTargetEncoder:
    """
    Leak-Free Out-of-Fold Universal Level Target & Frequency Encoder with Laplace Smoothing (SMOOTH=10.0).
    Encodes explicit string levels (including '__missing__') across all columns to extract CTGAN discrete likelihood priors.
    """
    def __init__(self, cols: Optional[List[str]] = None, smooth: float = 10.0, n_splits: int = 5, random_state: int = 42):
        self.cols = cols or [
            'daily_screen_time_hours', 'social_media_hours', 'gaming_hours',
            'work_study_hours', 'sleep_hours', 'notifications_per_day',
            'app_opens_per_day', 'weekend_screen_time', 'age',
            'gender', 'stress_level', 'academic_work_impact'
        ]
        self.smooth = smooth
        self.n_splits = n_splits
        self.random_state = random_state
        self.global_mean = 0.5
        self.te_mapping = {}
        self.freq_mapping = {}

    def _make_levels(self, df: pd.DataFrame) -> pd.DataFrame:
        cols_present = [c for c in self.cols if c in df.columns]
        return pd.DataFrame({
            c: df[c].astype(object).fillna('__missing__').astype(str).values
            for c in cols_present
        }, index=df.index)

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        X_out = X.copy()
        self.global_mean = float(y.mean())
        self.te_mapping = {}
        self.freq_mapping = {}

        levels_df = self._make_levels(X)
        cols_present = levels_df.columns.tolist()

        # Global mapping for validation / inference
        for col in cols_present:
            lvl_col = levels_df[col]
            stats = y.groupby(lvl_col).agg(['count', 'sum'])
            col_map = ((stats['sum'] + self.smooth * self.global_mean) / (stats['count'] + self.smooth)).to_dict()
            freq_map = (lvl_col.value_counts(normalize=True)).to_dict()
            self.te_mapping[col] = col_map
            self.freq_mapping[col] = freq_map

        # Internal K-Fold Out-of-Fold encoding for training
        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        for col in cols_present:
            oof_te = np.zeros(len(X), dtype=np.float32)
            oof_freq = np.zeros(len(X), dtype=np.float32)
            for tr_idx, val_idx in skf.split(X, y):
                lvl_tr = levels_df.iloc[tr_idx][col]
                lvl_vl = levels_df.iloc[val_idx][col]
                y_tr = y.iloc[tr_idx]

                tr_mean = float(y_tr.mean())
                stats_tr = y_tr.groupby(lvl_tr).agg(['count', 'sum'])
                map_tr = ((stats_tr['sum'] + self.smooth * tr_mean) / (stats_tr['count'] + self.smooth)).to_dict()
                freq_tr = (lvl_tr.value_counts(normalize=True)).to_dict()

                oof_te[val_idx] = lvl_vl.map(map_tr).fillna(tr_mean).values.astype(np.float32)
                oof_freq[val_idx] = lvl_vl.map(freq_tr).fillna(0.0).values.astype(np.float32)

            X_out[f'{col}_lvl_te'] = oof_te
            X_out[f'{col}_lvl_freq'] = oof_freq

        return X_out

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_out = X.copy()
        levels_df = self._make_levels(X)
        for col in levels_df.columns:
            if col in self.te_mapping:
                lvl_col = levels_df[col]
                X_out[f'{col}_lvl_te'] = lvl_col.map(self.te_mapping[col]).fillna(self.global_mean).values.astype(np.float32)
                X_out[f'{col}_lvl_freq'] = lvl_col.map(self.freq_mapping[col]).fillna(0.0).values.astype(np.float32)
        return X_out

# Backward compatibility alias
ValueLevelTargetEncoder = UniversalLevelTargetEncoder


class CompetitionSolver:
    def __init__(self, n_splits: int = 10, random_state: int = 42, use_neural_net: bool = True, n_estimators: Optional[int] = None):
        self.n_splits = n_splits
        self.random_state = random_state
        self.use_neural_net = use_neural_net
        self.n_estimators = n_estimators
        self.estimators = {}

    def cross_validate(self, X: pd.DataFrame, y: pd.Series) -> Tuple[np.ndarray, float]:
        """
        Executes a 10-fold Stratified CV loop targeting 'addicted_label'
        with strict leak-free local fold preprocessing, Value-Level Target Encoding,
        and 4-way heterogeneous modeling.
        """
        X = X.reset_index(drop=True)
        y = y.reset_index(drop=True)

        cv = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)

        num_models = 4 if self.use_neural_net else 3
        oof_preds_matrix = np.zeros((len(X), num_models))
        fold_scores = []

        self.fold_models = []
        self.fold_encoders = []

        for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
            print(f'Starting fold {fold + 1}/{self.n_splits}...', flush=True)
            X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
            X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

            # 1. Feature Engineering
            X_train_clean = preprocess_and_engineer(X_train)
            X_val_clean = preprocess_and_engineer(X_val)

            # 2. Out-of-Fold Universal Level Target & Frequency Encoding with Laplace smoothing
            te = UniversalLevelTargetEncoder(smooth=10.0, n_splits=5, random_state=self.random_state + fold)
            X_train_clean = te.fit_transform(X_train_clean, y_train)
            X_val_clean = te.transform(X_val_clean)

            # Categorical cols for encoding
            cat_cols = list(X_train_clean.select_dtypes(exclude=[np.number]).columns)

            # 3. Categorical Encoding localized to the fold
            encoders = {}
            for col in cat_cols:
                le = LabelEncoder()
                train_series = X_train_clean[col].fillna('Missing').astype(str)
                val_series = X_val_clean[col].fillna('Missing').astype(str)

                X_train_clean[col] = le.fit_transform(train_series)

                val_classes = list(set(val_series.tolist()))
                missing_classes = set(val_classes) - set(le.classes_)
                if missing_classes:
                    le.classes_ = np.append(le.classes_, list(missing_classes))
                X_val_clean[col] = le.transform(val_series)

                encoders[col] = le

            self.fold_encoders.append({
                'encoders': encoders,
                'target_encoder': te,
                'cat_cols': cat_cols
            })

            # 3. GBDT Hardware Acceleration
            lgb_params, xgb_params, cat_params = get_calibrated_model_params()
            if self.n_estimators is not None:
                lgb_params["n_estimators"] = self.n_estimators
                xgb_params["n_estimators"] = self.n_estimators
                cat_params["iterations"] = self.n_estimators
            lgb = LGBMClassifier(**lgb_params)
            xgb = XGBClassifier(**xgb_params)
            cat = CatBoostClassifier(**cat_params)

            lgb.fit(X_train_clean, y_train)
            xgb.fit(X_train_clean, y_train)
            cat.fit(X_train_clean, y_train)

            p_lgb = lgb.predict_proba(X_val_clean)[:, 1]
            p_xgb = xgb.predict_proba(X_val_clean)[:, 1]
            p_cat = cat.predict_proba(X_val_clean)[:, 1]

            fold_model_dict = {
                'lgb': lgb,
                'xgb': xgb,
                'cat': cat
            }

            if self.use_neural_net:
                nn_epochs = 1 if (self.n_estimators is not None and self.n_estimators < 10) else 8
                nn = DeepTabularClassifier(hidden_dim=128, num_blocks=2, epochs=nn_epochs, batch_size=4096)
                nn.fit(X_train_clean, y_train.values, cat_cols=cat_cols)
                p_nn = nn.predict_proba(X_val_clean)[:, 1]
                fold_model_dict['nn'] = nn
                blend_preds = (p_lgb + p_xgb + p_cat + p_nn) / 4.0
                oof_preds_matrix[val_idx, 3] = p_nn
            else:
                blend_preds = (p_lgb + p_xgb + p_cat) / 3.0

            self.fold_models.append(fold_model_dict)

            oof_preds_matrix[val_idx, 0] = p_lgb
            oof_preds_matrix[val_idx, 1] = p_xgb
            oof_preds_matrix[val_idx, 2] = p_cat

            fold_auc = roc_auc_score(y_val, blend_preds)
            fold_scores.append(fold_auc)

        mean_auc = np.mean(fold_scores)
        return oof_preds_matrix, mean_auc


def to_gauss_rank(ranks: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Transforms rank percentiles (0, 1) into standard Gaussian domain using inverse normal CDF (probit).
    Eliminates scale distortions between tree models (GBDT) and continuous MLP neural nets.
    """
    clipped = np.clip(ranks, eps, 1.0 - eps)
    return norm.ppf(clipped)


class LogisticStacker:
    """
    Nested Logistic Regression Stacker on Gauss-Rank Transformed Percentiles.
    Regularized at C=0.03 to prevent negative weight overfitting.
    """
    def __init__(self, C: float = 0.03, random_state: int = 42):
        self.C = C
        self.random_state = random_state
        self.model = LogisticRegression(C=self.C, max_iter=1000, random_state=self.random_state)
        self.coef_ = None
        self.intercept_ = None

    def fit(self, preds_matrix: np.ndarray, y: np.ndarray):
        self.model.fit(preds_matrix, y)
        self.coef_ = self.model.coef_[0]
        self.intercept_ = float(self.model.intercept_[0])
        return self

    def predict_proba(self, preds_matrix: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(preds_matrix)[:, 1]


def perform_ks_drift_screen(oof_rank: np.ndarray, test_rank: np.ndarray, threshold: float = 0.05) -> Tuple[bool, float]:
    """
    Kolmogorov-Smirnov two-sample test to detect rank distribution drift between OOF and Test sets.
    """
    stat, p_val = ks_2samp(oof_rank, test_rank)
    passed = stat <= threshold
    return passed, float(stat)


class NelderMeadRankStacker:
    """
    Direct Non-Parametric Nelder-Mead Rank-AUC Optimizer.
    Finds non-negative weights that directly maximize the ROC-AUC rank metric
    on out-of-fold probability matrices, eliminating calibration distortions.
    """
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.weights_ = None

    def _objective(self, unconstrained_weights: np.ndarray, preds_matrix: np.ndarray, y: np.ndarray) -> float:
        # Softmax projection ensures non-negativity and sum(w) = 1.0
        exp_w = np.exp(unconstrained_weights - np.max(unconstrained_weights))
        weights = exp_w / np.sum(exp_w)
        
        # Weighted rank blend
        blended = np.dot(preds_matrix, weights)
        return -roc_auc_score(y, blended)

    def fit(self, preds_matrix: np.ndarray, y: np.ndarray):
        n_models = preds_matrix.shape[1]
        init_weights = np.zeros(n_models)

        res = minimize(
            self._objective,
            x0=init_weights,
            args=(preds_matrix, y),
            method='Nelder-Mead',
            options={'maxiter': 500, 'xatol': 1e-4, 'fatol': 1e-5}
        )

        exp_w = np.exp(res.x - np.max(res.x))
        self.weights_ = exp_w / np.sum(exp_w)
        return self

    def predict_proba(self, preds_matrix: np.ndarray) -> np.ndarray:
        if self.weights_ is None:
            raise ValueError("NelderMeadRankStacker is not fitted yet.")
        return np.dot(preds_matrix, self.weights_)


class EnsembleBlender:
    """Legacy SLSQP Bounded Blender."""
    def __init__(self):
        self.weights_ = None

    def _objective(self, weights: np.ndarray, preds_matrix: np.ndarray, y: np.ndarray) -> float:
        blend_preds = np.dot(preds_matrix, weights)
        return -roc_auc_score(y, blend_preds)

    def fit(self, preds_matrix: np.ndarray, y: np.ndarray) -> np.ndarray:
        n_models = preds_matrix.shape[1]
        init_weights = np.ones(n_models) / n_models
        bounds = [(0.0, 1.0) for _ in range(n_models)]
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]

        result = minimize(
            self._objective,
            x0=init_weights,
            args=(preds_matrix, y),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'disp': False}
        )
        self.weights_ = result.x
        return self.weights_


def get_or_create_cloud_study(study_name: str = "s6e8_master_study") -> Any:
    """
    اتصال به دیتابیس لجر Cloud SQL و راه‌اندازی/بازیابی مطالعه اپتونا با قابلیت Warm-Start
    """
    import logging
    logger = logging.getLogger("OptunaCloudOrchestrator")

    db_user = os.getenv("CLOUD_SQL_USER", "postgres")
    db_pass = os.getenv("CLOUD_SQL_PASSWORD", "your-secure-password")
    db_name = os.getenv("CLOUD_SQL_DB", "optuna_ledger")
    db_host = os.getenv("CLOUD_SQL_HOST")  # آی‌پی داخلی اینستنس Cloud SQL

    try:
        import optuna
        from optuna.storages import RDBStorage
    except ImportError:
        logger.warning("⚠️ کتابخانه optuna نصب نیست. لطفاً آن را نصب کنید.")
        return None

    if not db_host:
        logger.warning("⚠️ آی‌پی Cloud SQL یافت نشد. در حال استفاده از مود InMemory موقت...")
        return optuna.create_study(direction="maximize")

    # ساخت URL اتصال استاندارد PostgreSQL برای اپتونا
    database_url = f"postgresql+psycopg2://{db_user}:{db_pass}@{db_host}:5432/{db_name}"

    # پیکربندی مخزن داده رابطه‌ای (RDBStorage)
    storage = RDBStorage(
        url=database_url,
        engine_kwargs={
            "pool_size": 10,
            "max_overflow": 20,
            "pool_pre_ping": True  # بررسی زنده بودن کانکشن قبل از ارسال کوئری
        }
    )

    logger.info(f"🔄 در حال بررسی وضعیت مطالعه [{study_name}] روی لجر ابری...")
    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        storage=storage,
        load_if_exists=True    # 👈 کلید طلایی Warm-Start: بازیابی خودکار از ادامه تریال‌ها
    )

    logger.info(f"✅ مطالعه با موفقیت بازیابی/ایجاد شد. تعداد تریال‌های ثبت شده فعلی: {len(study.trials)}")
    return study


