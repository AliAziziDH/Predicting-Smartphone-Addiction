import os
import json
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional, List
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from scipy.stats import ks_2samp, norm, rankdata
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from scipy.optimize import minimize

from src.model.formulation import preprocess_and_engineer
from src.model.neural_tabular import DeepTabularClassifier


# Calibrated, highly regularized GBDT configurations
LGBM_PARAMS = {
    "objective": "binary",
    "metric": "auc",
    "boosting_type": "gbdt",
    "n_estimators": 2500,
    "learning_rate": 0.02,
    "num_leaves": 63,
    "max_depth": -1,
    "min_child_samples": 100,
    "subsample": 0.85,
    "colsample_bytree": 0.70,
    "reg_alpha": 0.1,
    "reg_lambda": 3.0,
    "random_state": 42,
    "verbose": -1,
    "n_jobs": -1
}

XGB_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "n_estimators": 2500,
    "learning_rate": 0.02,
    "max_depth": 6,
    "min_child_weight": 10,
    "subsample": 0.85,
    "colsample_bytree": 0.65,
    "reg_alpha": 0.5,
    "reg_lambda": 5.0,
    "random_state": 42,
    "tree_method": "hist",
    "n_jobs": -1
}

CAT_PARAMS = {
    "loss_function": "Logloss",
    "eval_metric": "AUC",
    "iterations": 2200,
    "learning_rate": 0.025,
    "depth": 6,
    "l2_leaf_reg": 5.0,
    "bootstrap_type": "Bernoulli",
    "subsample": 0.85,
    "random_state": 42,
    "verbose": False,
    "thread_count": -1
}


def get_calibrated_model_params() -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """Dynamically routes parameters to CUDA GPU if available and loads Optuna tuned parameters."""
    lgb_p = LGBM_PARAMS.copy()
    xgb_p = XGB_PARAMS.copy()
    cat_p = CAT_PARAMS.copy()

    tuned_path = os.path.join(os.getcwd(), "models", "best_gbdt_params.json")
    if os.path.exists(tuned_path):
        try:
            with open(tuned_path, "r") as f:
                tuned_data = json.load(f)
            if "lgb_params" in tuned_data:
                lgb_p.update(tuned_data["lgb_params"])
            if "xgb_params" in tuned_data:
                xgb_p.update(tuned_data["xgb_params"])
        except Exception as e:
            pass

    try:
        import torch
        has_cuda = torch.cuda.is_available()
    except Exception:
        has_cuda = False

    if has_cuda:
        xgb_p["device"] = "cuda"
        cat_p["task_type"] = "GPU"
    else:
        xgb_p["device"] = "cpu"
        cat_p["task_type"] = "CPU"

    return lgb_p, xgb_p, cat_p


class DiscreteCategoricalTargetEncoder:
    """
    Leak-Free Out-of-Fold Target & Frequency Encoder strictly for discrete categoricals.
    Applies Laplace smoothing (smooth=20.0) to prevent overfitting on low-cardinality groups.
    """
    def __init__(self, cat_cols: Optional[List[str]] = None, smooth: float = 20.0, n_splits: int = 5, random_state: int = 42):
        self.cat_cols = cat_cols or ['gender', 'stress_level', 'academic_work_impact']
        self.smooth = smooth
        self.n_splits = n_splits
        self.random_state = random_state
        self.cat_pairs = [
            ('gender', 'stress_level'),
            ('gender', 'academic_work_impact'),
            ('stress_level', 'academic_work_impact'),
        ]
        self.global_mean = 0.5
        self.te_mapping = {}
        self.freq_mapping = {}

    def _make_discrete_levels(self, df: pd.DataFrame) -> pd.DataFrame:
        cols_present = [c for c in self.cat_cols if c in df.columns]
        levels_dict = {
            c: df[c].fillna('__missing__').astype(str).values
            for c in cols_present
        }
        for c1, c2 in self.cat_pairs:
            if c1 in df.columns and c2 in df.columns:
                v1 = df[c1].fillna('__missing__').astype(str).values
                v2 = df[c2].fillna('__missing__').astype(str).values
                levels_dict[f'{c1}_{c2}'] = (v1 + '_' + v2)
        return pd.DataFrame(levels_dict, index=df.index)

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        X_out = X.copy()
        self.global_mean = float(y.mean())
        self.te_mapping = {}
        self.freq_mapping = {}

        levels_df = self._make_discrete_levels(X)
        cols_present = levels_df.columns.tolist()
        y_arr = y.to_numpy(dtype=np.float64)

        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        splits = list(skf.split(X, y))

        for col in cols_present:
            codes, uniques = pd.factorize(levels_df[col].values)
            n_uniques = len(uniques)

            # Global mapping
            counts = np.bincount(codes, minlength=n_uniques)
            sums = np.bincount(codes, weights=y_arr, minlength=n_uniques)
            smoothed_global = (sums + self.smooth * self.global_mean) / (counts + self.smooth)
            freqs_global = counts / len(codes)

            self.te_mapping[col] = dict(zip(uniques, smoothed_global))
            self.freq_mapping[col] = dict(zip(uniques, freqs_global))

            # Fast Out-of-Fold computation
            oof_te = np.zeros(len(X), dtype=np.float32)
            oof_freq = np.zeros(len(X), dtype=np.float32)

            for tr_idx, val_idx in splits:
                tr_codes = codes[tr_idx]
                tr_y = y_arr[tr_idx]
                tr_mean = float(tr_y.mean())

                tr_counts = np.bincount(tr_codes, minlength=n_uniques)
                tr_sums = np.bincount(tr_codes, weights=tr_y, minlength=n_uniques)

                mask_observed = tr_counts > 0
                tr_smoothed = np.full(n_uniques, tr_mean, dtype=np.float32)
                tr_smoothed[mask_observed] = (tr_sums[mask_observed] + self.smooth * tr_mean) / (tr_counts[mask_observed] + self.smooth)
                tr_freqs = (tr_counts / len(tr_codes)).astype(np.float32)

                val_codes = codes[val_idx]
                oof_te[val_idx] = tr_smoothed[val_codes]
                oof_freq[val_idx] = tr_freqs[val_codes]

            X_out[f'{col}_te'] = oof_te
            X_out[f'{col}_freq'] = oof_freq

        return X_out

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_out = X.copy()
        levels_df = self._make_discrete_levels(X)
        for col in levels_df.columns:
            if col in self.te_mapping:
                lvl_col = levels_df[col]
                X_out[f'{col}_te'] = lvl_col.map(self.te_mapping[col]).fillna(self.global_mean).values.astype(np.float32)
                X_out[f'{col}_freq'] = lvl_col.map(self.freq_mapping[col]).fillna(0.0).values.astype(np.float32)
        return X_out


# Backward compatibility aliases
UniversalLevelTargetEncoder = DiscreteCategoricalTargetEncoder
ValueLevelTargetEncoder = DiscreteCategoricalTargetEncoder


class CompetitionSolver:
    def __init__(self, n_splits: int = 10, random_state: int = 42, use_neural_net: bool = True, n_estimators: Optional[int] = None):
        self.n_splits = n_splits
        self.random_state = random_state
        self.use_neural_net = use_neural_net
        self.n_estimators = n_estimators
        self.fold_models = []
        self.fold_encoders = []

    def cross_validate(self, X: pd.DataFrame, y: pd.Series) -> Tuple[np.ndarray, float]:
        """
        Executes a leak-free 10-fold Stratified CV loop with clean feature engineering,
        discrete target encoding, and diverse model ensembling.
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
            X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
            X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

            # 1. Clean Feature Engineering
            X_train_clean = preprocess_and_engineer(X_train)
            X_val_clean = preprocess_and_engineer(X_val)

            # 2. Leak-Free Discrete Target & Frequency Encoding
            te = DiscreteCategoricalTargetEncoder(smooth=20.0, n_splits=5, random_state=self.random_state + fold)
            X_train_clean = te.fit_transform(X_train_clean, y_train)
            X_val_clean = te.transform(X_val_clean)

            # 3. Categorical Label Encoding localized to the fold
            cat_cols = list(X_train_clean.select_dtypes(exclude=[np.number]).columns)
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

            # 4. GBDT Training
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

            fold_model_dict = {'lgb': lgb, 'xgb': xgb, 'cat': cat}

            if self.use_neural_net:
                nn_epochs = 1 if (self.n_estimators is not None and self.n_estimators < 10) else 6
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

        mean_auc = float(np.mean(fold_scores))
        return oof_preds_matrix, mean_auc


def to_gauss_rank(ranks: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Transforms rank percentiles (0, 1) into standard Gaussian domain using probit."""
    clipped = np.clip(ranks, eps, 1.0 - eps)
    return norm.ppf(clipped)


class LogisticStacker:
    """Logistic Regression Stacker on Gauss-Rank Transformed Percentiles."""
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


class EnsembleBlender:
    """
    SLSQP Bounded Convex Optimization Blender.
    Finds optimal non-negative weights (sum=1.0, w_i >= 0) to maximize ROC-AUC.
    """
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


class NelderMeadRankStacker:
    """Non-Parametric Nelder-Mead Rank-AUC Optimizer with Softmax Projection."""
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.weights_ = None

    def _objective(self, unconstrained_weights: np.ndarray, preds_matrix: np.ndarray, y: np.ndarray) -> float:
        exp_w = np.exp(unconstrained_weights - np.max(unconstrained_weights))
        weights = exp_w / np.sum(exp_w)
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


class TwoStageHybridStacker:
    """Two-Stage Robust Blending Stacker."""
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.tree_weights_ = None
        self.alpha_ = None

    def _to_rank(self, p: np.ndarray) -> np.ndarray:
        return (rankdata(p) - 0.5) / len(p)

    def fit(self, oof_lgb: np.ndarray, oof_cat: np.ndarray, oof_xgb: np.ndarray, oof_nn: np.ndarray, y: np.ndarray):
        r_lgb = self._to_rank(oof_lgb)
        r_cat = self._to_rank(oof_cat)
        r_xgb = self._to_rank(oof_xgb)
        r_nn = self._to_rank(oof_nn)

        def objective(params):
            w1, w2, w3, alpha = params
            w_sum = w1 + w2 + w3 + 1e-8
            w1_n, w2_n, w3_n = w1 / w_sum, w2 / w_sum, w3 / w_sum

            r_tree = w1_n * r_lgb + w2_n * r_cat + w3_n * r_xgb
            p_final = alpha * r_tree + (1.0 - alpha) * r_nn
            return -roc_auc_score(y, p_final)

        init_params = [0.33, 0.33, 0.33, 0.85]
        bounds = [(0, 1), (0, 1), (0, 1), (0, 1)]
        res = minimize(objective, init_params, method='Nelder-Mead', bounds=bounds)

        w1, w2, w3, alpha = res.x
        w_sum = w1 + w2 + w3 + 1e-8
        self.tree_weights_ = np.array([w1 / w_sum, w2 / w_sum, w3 / w_sum], dtype=np.float32)
        self.alpha_ = float(alpha)
        return self

    def predict_proba(self, p_lgb: np.ndarray, p_cat: np.ndarray, p_xgb: np.ndarray, p_nn: np.ndarray) -> np.ndarray:
        if self.tree_weights_ is None:
            raise ValueError("TwoStageHybridStacker is not fitted yet.")
        r_lgb = self._to_rank(p_lgb)
        r_cat = self._to_rank(p_cat)
        r_xgb = self._to_rank(p_xgb)
        r_nn = self._to_rank(p_nn)

        r_tree = self.tree_weights_[0] * r_lgb + self.tree_weights_[1] * r_cat + self.tree_weights_[2] * r_xgb
        p_final = self.alpha_ * r_tree + (1.0 - self.alpha_) * r_nn
        return p_final
