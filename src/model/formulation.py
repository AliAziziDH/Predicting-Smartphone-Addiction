from typing import List, Literal, Optional, Union
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, model_validator, ConfigDict

class UserBehaviorInput(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    age: Optional[float] = Field(None, ge=10, le=120)
    gender: Optional[str] = Field(None)
    daily_screen_time_hours: Optional[float] = Field(None, ge=0.0, le=24.0)
    social_media_hours: Optional[float] = Field(None, ge=0.0, le=24.0)
    gaming_hours: Optional[float] = Field(None, ge=0.0, le=24.0)
    work_study_hours: Optional[float] = Field(None, ge=0.0, le=24.0)
    sleep_hours: Optional[float] = Field(None, ge=0.0, le=24.0)
    notifications_per_day: Optional[float] = Field(None, ge=0.0)
    app_opens_per_day: Optional[float] = Field(None, ge=0.0)
    weekend_screen_time: Optional[float] = Field(None, ge=0.0, le=48.0)
    stress_level: Optional[str] = Field(None)
    academic_work_impact: Optional[str] = Field(None)

    @model_validator(mode='after')
    def check_sub_durations(self):
        if self.social_media_hours is not None and self.daily_screen_time_hours is not None:
            if not np.isnan(self.social_media_hours) and not np.isnan(self.daily_screen_time_hours):
                if self.social_media_hours > self.daily_screen_time_hours:
                    pass

        if self.gaming_hours is not None and self.daily_screen_time_hours is not None:
            if not np.isnan(self.gaming_hours) and not np.isnan(self.daily_screen_time_hours):
                if self.gaming_hours > self.daily_screen_time_hours:
                    pass

        if self.work_study_hours is not None and self.daily_screen_time_hours is not None:
            if not np.isnan(self.work_study_hours) and not np.isnan(self.daily_screen_time_hours):
                if self.work_study_hours > self.daily_screen_time_hours:
                    pass

        return self

def preprocess_and_engineer(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validates input and engineers behavioral features and grid-frequency counts.
    Preserves native NaN propagation for optimal GBDT tree branching.
    """
    df = df.copy()

    # 1. Pydantic Validation
    records = df.to_dict(orient='records')
    validated_records = []
    for r in records:
        clean_r = {}
        for k, v in r.items():
            if pd.isna(v):
                clean_r[k] = None
            else:
                clean_r[k] = v

        validated = UserBehaviorInput(**clean_r)
        validated_records.append(validated.model_dump())

    df_clean = pd.DataFrame(validated_records)

    # Preserve any extra dynamic/candidate columns passed into preprocess_and_engineer
    extra_cols = [c for c in df.columns if c not in UserBehaviorInput.model_fields and c != 'id']
    for c in extra_cols:
        df_clean[c] = df[c].values

    # 2. Feature Engineering
    eps = 1e-9

    # 1. Residual Screen Time (Other Screen) with native NaN propagation
    df_clean['other_screen'] = df_clean['daily_screen_time_hours'] - (
        df_clean['social_media_hours'] + df_clean['gaming_hours'] + df_clean['work_study_hours']
    )

    # 2. Life Residual / Unaccounted Hours (24‑Hour Constraint) with native NaN propagation
    df_clean['unaccounted_hours'] = 24.0 - (
        df_clean['daily_screen_time_hours'] + df_clean['work_study_hours'] + df_clean['sleep_hours']
    )

    # 3. Behavioral High‑Risk Ratios (handle NaN without eps for NaN propagation)
    df_clean['gaming_to_screen_ratio'] = np.where(
        np.isnan(df_clean['daily_screen_time_hours']),
        np.nan,
        df_clean['gaming_hours'] / (df_clean['daily_screen_time_hours'] + eps)
    )
    df_clean['social_to_screen_ratio'] = np.where(
        np.isnan(df_clean['daily_screen_time_hours']),
        np.nan,
        df_clean['social_media_hours'] / (df_clean['daily_screen_time_hours'] + eps)
    )
    df_clean['screen_to_sleep_ratio'] = np.where(
        np.isnan(df_clean['sleep_hours']),
        np.nan,
        df_clean['daily_screen_time_hours'] / (df_clean['sleep_hours'] + eps)
    )

    # 4. Per‑hour activity rates (avoid division by zero)
    df_clean['notifications_per_hour'] = df_clean['notifications_per_day'] / (df_clean['daily_screen_time_hours'] + eps)
    df_clean['app_opens_per_hour'] = df_clean['app_opens_per_day'] / (df_clean['daily_screen_time_hours'] + eps)

    # 5. Weekend screen‑time proportion
    df_clean['weekend_screen_time_ratio'] = df_clean['weekend_screen_time'] / (df_clean['daily_screen_time_hours'] + eps)

    # 6. Sleep deficit relative to 8 h target
    df_clean['sleep_deficit'] = 8.0 - df_clean['sleep_hours']

    # 7. Total activity hours (social + gaming + work + sleep)
    df_clean['total_activity_hours'] = df_clean[[
        'social_media_hours', 'gaming_hours', 'work_study_hours', 'sleep_hours'
    ]].sum(axis=1)

    # 8. Screen‑time proportion of total activity
    df_clean['screen_time_ratio_total'] = df_clean['daily_screen_time_hours'] / (df_clean['total_activity_hours'] + eps)

    # 9. Insomnia Compulsive Habit: app opens per sleep hour (handle NaN gracefully)
    df_clean['sleep_app_opens_ratio'] = np.where(
        np.isnan(df_clean['sleep_hours']),
        np.nan,
        df_clean['app_opens_per_day'] / (df_clean['sleep_hours'] + eps)
    )

    # 10. Unsupervised Grid-Frequency / Relative Density Encoding on synthetic rounded features
    grid_cols = ['app_opens_per_day', 'notifications_per_day', 'daily_screen_time_hours', 'weekend_screen_time', 'age']
    for col in grid_cols:
        if col in df_clean.columns:
            freq_map = df_clean[col].value_counts(normalize=True, dropna=True).to_dict()
            df_clean[f'{col}_freq'] = df_clean[col].map(freq_map)

    # 11. Productive Work Shield (shields high-screen productive workers from False Positives)
    df_clean['productive_work_ratio'] = np.where(
        np.isnan(df_clean['daily_screen_time_hours']),
        np.nan,
        df_clean['work_study_hours'] / (df_clean['daily_screen_time_hours'] + eps)
    )

    # 12. Work-Adjusted Screen Load (pure non-work screen strain relative to sleep)
    df_clean['work_adjusted_screen_load'] = np.where(
        np.isnan(df_clean['sleep_hours']) | np.isnan(df_clean['daily_screen_time_hours']),
        np.nan,
        (df_clean['daily_screen_time_hours'] - df_clean['work_study_hours']) / (df_clean['sleep_hours'] + eps)
    )

    # 13. Compulsive Checking Velocity (internal compulsion intensity)
    df_clean['compulsive_checking_velocity'] = np.where(
        np.isnan(df_clean['daily_screen_time_hours']) | np.isnan(df_clean['sleep_hours']),
        np.nan,
        (df_clean['notifications_per_day'] / (df_clean['app_opens_per_day'] + 1.0)) * 
        (df_clean['daily_screen_time_hours'] / np.maximum(eps, 24.0 - df_clean['sleep_hours']))
    )

    # 14. Circadian Dopamine Strain (polynomial nighttime dopamine penalty)
    social_gaming = df_clean['social_media_hours'].fillna(0.0) + df_clean['gaming_hours'].fillna(0.0)
    sleep_def = np.maximum(0.0, 8.0 - df_clean['sleep_hours'].fillna(8.0))
    df_clean['circadian_dopamine_strain'] = np.where(
        np.isnan(df_clean['sleep_hours']),
        np.nan,
        sleep_def * np.power(np.maximum(0.0, social_gaming), 1.5) / (df_clean['sleep_hours'] + 1.0)
    )

    # 15. Synthetic Time Budget Deficit & Violation Indicator (captures CTGAN generator artifact)
    claimed_activities = df_clean['social_media_hours'].fillna(0.0) + df_clean['gaming_hours'].fillna(0.0) + df_clean['work_study_hours'].fillna(0.0)
    df_clean['synthetic_budget_deficit'] = np.where(
        np.isnan(df_clean['daily_screen_time_hours']),
        np.nan,
        claimed_activities - df_clean['daily_screen_time_hours']
    )
    df_clean['synthetic_budget_violation'] = np.where(
        np.isnan(df_clean['daily_screen_time_hours']),
        np.nan,
        (claimed_activities > df_clean['daily_screen_time_hours']).astype(np.float32)
    )

    # 16. Multi-Way Demographic Joint Profile Frequency
    if 'gender' in df_clean.columns and 'stress_level' in df_clean.columns and 'academic_work_impact' in df_clean.columns:
        joint_key = (
            df_clean['gender'].astype(str) + '_' + 
            df_clean['stress_level'].astype(str) + '_' + 
            df_clean['academic_work_impact'].astype(str)
        )
        joint_freq_map = joint_key.value_counts(normalize=True).to_dict()
        df_clean['joint_profile_freq'] = joint_key.map(joint_freq_map).astype(np.float32)

    # Helper for safe float series access across sparse test dataframes
    def _safe_col(col_name: str, default: float = 0.0) -> pd.Series:
        if col_name in df_clean.columns:
            return pd.to_numeric(df_clean[col_name], errors='coerce').fillna(default).astype(np.float32)
        return pd.Series(default, index=df_clean.index, dtype=np.float32)

    # 17. Inter-Session Arrival Time Burstiness (B_ISAT) - AI Studio Prescription
    scr_hrs = _safe_col('daily_screen_time_hours', default=1.0)
    notifs = _safe_col('notifications_per_day', default=1.0)
    app_ops = _safe_col('app_opens_per_day', default=0.0)
    df_clean['B_ISAT'] = ((app_ops / (scr_hrs + eps)) / (notifs + eps)).astype(np.float32)

    # 18. Attention Fragmentation Entropy (H_AF) - AI Studio Prescription
    soc_hrs = _safe_col('social_media_hours', default=0.0)
    gam_hrs = _safe_col('gaming_hours', default=0.0)
    wrk_hrs = _safe_col('work_study_hours', default=0.0)
    tot_act_time = soc_hrs + gam_hrs + wrk_hrs + eps
    p_soc = soc_hrs / tot_act_time
    p_gam = gam_hrs / tot_act_time
    p_wrk = wrk_hrs / tot_act_time
    df_clean['H_AF'] = -(p_soc * np.log(p_soc + eps) + p_gam * np.log(p_gam + eps) + p_wrk * np.log(p_wrk + eps)).astype(np.float32)

    # 19. Chronobiological Dopamine-to-Utility Ratio (CDUR) - AI Studio Prescription
    dopamine_hrs = soc_hrs + gam_hrs
    utility_hrs = wrk_hrs
    slp_hrs = _safe_col('sleep_hours', default=8.0)
    chrono_penalty = 24.0 / (slp_hrs + eps)
    df_clean['CDUR'] = ((dopamine_hrs / (utility_hrs + eps)) * chrono_penalty).astype(np.float32)

    # 20. Dopamine Chasing Index - AI Studio Synergy
    leisure_hrs = soc_hrs + gam_hrs
    sigmoid_leisure = 1.0 / (1.0 + np.exp(-np.clip(leisure_hrs - 3.0, -10.0, 10.0)))
    df_clean['dopamine_chasing_index'] = (np.log1p(app_ops / (notifs + 1.0)) * sigmoid_leisure).astype(np.float32)

    # 21. Compulsive Checking Density - AI Studio Synergy
    awake_hrs = np.maximum(eps, 24.0 - slp_hrs)
    df_clean['compulsive_checking_density'] = (((app_ops + notifs) / awake_hrs) * (1.0 / (scr_hrs + eps))).astype(np.float32)

    # 22. Decimal Lattice Features (Sub-unit continuous position and first decimal digit CTGAN artifacts)
    frac_cols = ['daily_screen_time_hours', 'social_media_hours', 'gaming_hours', 'work_study_hours', 'sleep_hours', 'weekend_screen_time']
    for c in frac_cols:
        if c in df_clean.columns:
            val = pd.to_numeric(df_clean[c], errors='coerce').to_numpy(dtype=np.float64)
            df_clean[f'frac_{c}'] = np.where(np.isnan(val), np.nan, val - np.floor(val)).astype(np.float32)
            df_clean[f'd1_{c}'] = np.where(np.isnan(val), np.nan, np.floor(val * 10.0) % 10.0).astype(np.float32)

    # 23. Missing Pattern Count & Explicit Missingness Indicators
    raw_num_cols = ['age', 'daily_screen_time_hours', 'social_media_hours', 'gaming_hours', 'work_study_hours', 'sleep_hours', 'notifications_per_day', 'app_opens_per_day', 'weekend_screen_time']
    existing_raw = [c for c in raw_num_cols if c in df_clean.columns]
    df_clean['missing_features_count'] = df_clean[existing_raw].isna().sum(axis=1).astype(np.float32)

    # 24. ClickHouse Top Mathematical Interactions (Wave 10)
    df_clean['st_mul_sm'] = (scr_hrs * soc_hrs).astype(np.float32)
    df_clean['st_div_awake'] = (scr_hrs / (awake_hrs + eps)).astype(np.float32)
    df_clean['st_sub_gm'] = (scr_hrs - gam_hrs).astype(np.float32)
    df_clean['st_risk_boundary'] = ((scr_hrs - 5.5) * (soc_hrs + gam_hrs)).astype(np.float32)
    df_clean['screen_intensity'] = ((app_ops * scr_hrs) / (awake_hrs + eps)).astype(np.float32)

    # 25. BigQuery Group Cohort Residuals (Wave 10)
    if 'age' in df_clean.columns and 'gender' in df_clean.columns:
        cohort_key = df_clean['age'].astype(str) + '_' + df_clean['gender'].astype(str)
        for target_c, target_arr in [('st', scr_hrs), ('sm', soc_hrs)]:
            s = pd.Series(target_arr)
            c_mean = s.groupby(cohort_key).transform('mean').to_numpy(dtype=np.float32)
            c_std = s.groupby(cohort_key).transform('std').fillna(1.0).to_numpy(dtype=np.float32)
            df_clean[f'cohort_diff_{target_c}'] = (target_arr - c_mean).astype(np.float32)
            df_clean[f'cohort_zscore_{target_c}'] = ((target_arr - c_mean) / (c_std + eps)).astype(np.float32)

    # 26. High-Performance Memory Downcasting (reduces RAM from 1.2GB to <300MB on 690k rows)
    for col in df_clean.select_dtypes(include=['float64']).columns:
        df_clean[col] = df_clean[col].astype(np.float32)
    for col in df_clean.select_dtypes(include=['int64']).columns:
        df_clean[col] = pd.to_numeric(df_clean[col], downcast='integer')

    return df_clean

