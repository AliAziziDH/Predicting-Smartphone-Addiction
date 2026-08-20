"""
Master Benchmark Gate for Wave 10 Candidates.
Integrates the 5 Intelligence Tools:
1. ClickHouse Top Interactions (st * sm, st / (24 - sl), (st - 5.5) * (sm + gm))
2. BigQuery Cohort Residuals (st - mean[age, gender], sm - mean[age, gender])
3. Discrete Categorical 2-Way TE (gender x stress, gender x impact, stress x impact)
4. 3-Way Model Zoo (LGBM, XGBoost, CatBoost)
5. Gauss-Rank SLSQP/Logistic Meta-Stacker
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from scipy.stats import rankdata, norm
from sklearn.linear_model import LogisticRegression

# 1. Load Data (30k stratified proxy rows)
print("📊 [BENCHMARK MASTER] Loading 30,000 stratified proxy rows...")
df_full = pd.read_csv("data/train.csv")
np.random.seed(42)
sample_idx = df_full.groupby("addicted_label", group_keys=False).apply(
    lambda x: x.sample(n=15000, random_state=42)
).index
df = df_full.loc[sample_idx].reset_index(drop=True)
y = df["addicted_label"].values

from src.model.formulation import preprocess_and_engineer

# Base Wave 9 features
X_base = preprocess_and_engineer(df).drop(columns=["id", "addicted_label"], errors="ignore")
for col in X_base.select_dtypes(include=["object", "string"]).columns:
    X_base[col] = pd.factorize(X_base[col].fillna("__missing__"))[0]

# 2. Add Master Wave 10 Candidate Features (ClickHouse + BigQuery)
X_wave10 = X_base.copy()

# A. ClickHouse Top Mathematical Interactions
st = df["daily_screen_time_hours"].fillna(df["daily_screen_time_hours"].median())
sl = df["sleep_hours"].fillna(df["sleep_hours"].median())
sm = df["social_media_hours"].fillna(0)
gm = df["gaming_hours"].fillna(0)
ao = df["app_opens_per_day"].fillna(df["app_opens_per_day"].median())

X_wave10["st_mul_sm"] = st * sm
X_wave10["st_div_awake"] = st / (24.0 - sl + 1e-4)
X_wave10["st_sub_gm"] = st - gm
X_wave10["st_risk_boundary"] = (st - 5.5) * (sm + gm)
X_wave10["screen_intensity"] = (ao * st) / (24.0 - sl + 1e-4)

# B. BigQuery Cohort Residuals (age + gender)
age_gen = df["age"].astype(str) + "_" + df["gender"].astype(str)
for num_col in ["daily_screen_time_hours", "social_media_hours"]:
    vals = df[num_col].fillna(df[num_col].median())
    cohort_mean = vals.groupby(age_gen).transform("mean")
    cohort_std = vals.groupby(age_gen).transform("std").fillna(1.0)
    X_wave10[f"cohort_diff_{num_col}"] = vals - cohort_mean
    X_wave10[f"cohort_zscore_{num_col}"] = (vals - cohort_mean) / (cohort_std + 1e-4)

raw_cols = [c for c in df.columns if c not in ["id", "addicted_label"]]
cat_pairs = [
    ("gender", "stress_level"),
    ("gender", "academic_work_impact"),
    ("stress_level", "academic_work_impact"),
]

def apply_full_te(train_df, val_df, cols, cat_pairs, target, smooth=10.0):
    tr_te = pd.DataFrame(index=train_df.index)
    va_te = pd.DataFrame(index=val_df.index)
    global_mean = target.mean()
    
    # 1-way Universal Level TE
    for c in cols:
        tr_vals = train_df[c].astype(str)
        va_vals = val_df[c].astype(str)
        stats = pd.DataFrame({"cat": tr_vals, "target": target}).groupby("cat")["target"].agg(["count", "mean"])
        smoothed = (stats["count"] * stats["mean"] + smooth * global_mean) / (stats["count"] + smooth)
        te_map = smoothed.to_dict()
        tr_te[f"te_{c}"] = tr_vals.map(te_map).fillna(global_mean)
        va_te[f"te_{c}"] = va_vals.map(te_map).fillna(global_mean)
        
        tr_freq = tr_vals.map(tr_vals.value_counts(normalize=True))
        va_freq = va_vals.map(tr_vals.value_counts(normalize=True)).fillna(0)
        tr_te[f"freq_{c}"] = tr_freq
        va_te[f"freq_{c}"] = va_freq
        
    # Discrete Categorical Interactions
    for c1, c2 in cat_pairs:
        tr_vals = train_df[c1].astype(str) + "_" + train_df[c2].astype(str)
        va_vals = val_df[c1].astype(str) + "_" + val_df[c2].astype(str)
        stats = pd.DataFrame({"cat": tr_vals, "target": target}).groupby("cat")["target"].agg(["count", "mean"])
        smoothed = (stats["count"] * stats["mean"] + smooth * global_mean) / (stats["count"] + smooth)
        te_map = smoothed.to_dict()
        tr_te[f"te_{c1}_{c2}"] = tr_vals.map(te_map).fillna(global_mean)
        va_te[f"te_{c1}_{c2}"] = va_vals.map(te_map).fillna(global_mean)
        
    return tr_te, va_te

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("\n🚀 [MASTER BENCHMARK] Evaluating 3-Way Model Zoo on Candidate Wave 10...")
oof_lgb = np.zeros(len(df))
oof_xgb = np.zeros(len(df))
oof_cat = np.zeros(len(df))

for fold, (trn_idx, val_idx) in enumerate(skf.split(df, y)):
    tr_te, va_te = apply_full_te(df.iloc[trn_idx], df.iloc[val_idx], raw_cols, cat_pairs, y[trn_idx])
    
    X_tr = pd.concat([X_wave10.iloc[trn_idx], tr_te], axis=1)
    X_va = pd.concat([X_wave10.iloc[val_idx], va_te], axis=1)
    
    # LGBM
    lgb = LGBMClassifier(n_estimators=350, learning_rate=0.035, num_leaves=127, subsample=0.85, colsample_bytree=0.85, random_state=42, n_jobs=4, verbose=-1)
    lgb.fit(X_tr, y[trn_idx])
    oof_lgb[val_idx] = lgb.predict_proba(X_va)[:, 1]
    
    # XGBoost
    xgb = XGBClassifier(n_estimators=350, learning_rate=0.035, max_depth=7, subsample=0.85, colsample_bytree=0.85, random_state=42, n_jobs=4, eval_metric="auc")
    xgb.fit(X_tr, y[trn_idx])
    oof_xgb[val_idx] = xgb.predict_proba(X_va)[:, 1]
    
    # CatBoost
    cat = CatBoostClassifier(iterations=350, learning_rate=0.04, depth=7, random_seed=42, verbose=0, thread_count=4)
    cat.fit(X_tr, y[trn_idx])
    oof_cat[val_idx] = cat.predict_proba(X_va)[:, 1]

auc_lgb = roc_auc_score(y, oof_lgb)
auc_xgb = roc_auc_score(y, oof_xgb)
auc_cat = roc_auc_score(y, oof_cat)

# Rank-Gauss Meta-Stacking
def to_gauss(p):
    r = rankdata(p) / (len(p) + 1.0)
    return norm.ppf(np.clip(r, 1e-5, 1 - 1e-5))

meta_X = np.column_stack([to_gauss(oof_lgb), to_gauss(oof_xgb), to_gauss(oof_cat)])
oof_meta = np.zeros(len(df))

for trn_idx, val_idx in skf.split(df, y):
    meta_clf = LogisticRegression(C=0.03, max_iter=200)
    meta_clf.fit(meta_X[trn_idx], y[trn_idx])
    oof_meta[val_idx] = meta_clf.predict_proba(meta_X[val_idx])[:, 1]

auc_meta = roc_auc_score(y, oof_meta)

print("=" * 60)
print(f"• Baseline Wave 9 Ensemble AUC: 0.95826")
print(f"• LightGBM Candidate AUC:      {auc_lgb:.5f}")
print(f"• XGBoost Candidate AUC:       {auc_xgb:.5f}")
print(f"• CatBoost Candidate AUC:      {auc_cat:.5f}")
print(f"🏆 MASTER WAVE 10 META-STACKER: {auc_meta:.5f}")
print(f"📈 TOTAL EMPIRICAL UPLIFT:      +{auc_meta - 0.95826:.5f} AUC!")
print("=" * 60)
