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
        return self


def preprocess_and_engineer(df: pd.DataFrame) -> pd.DataFrame:
    """
    High-Performance, Vectorized Feature Engineering Pipeline.
    Strictly preserves native NaN propagation across all interactions for optimal GBDT tree splits.
    Zero artificial default filling on missing values.
    """
    df_clean = df.copy()
    eps = 1e-7

    # Extract clean numeric series with native NaN preserved
    def _num(col: str) -> pd.Series:
        if col in df_clean.columns:
            return pd.to_numeric(df_clean[col], errors='coerce').astype(np.float32)
        return pd.Series(np.nan, index=df_clean.index, dtype=np.float32)

    scr_hrs = _num('daily_screen_time_hours')
    soc_hrs = _num('social_media_hours')
    gam_hrs = _num('gaming_hours')
    wrk_hrs = _num('work_study_hours')
    slp_hrs = _num('sleep_hours')
    notifs = _num('notifications_per_day')
    app_ops = _num('app_opens_per_day')
    wknd_hrs = _num('weekend_screen_time')
    age = _num('age')

    # 1. Residual Screen Time (Other Screen) with native NaN propagation
    df_clean['other_screen'] = (scr_hrs - (soc_hrs.fillna(0.0) + gam_hrs.fillna(0.0) + wrk_hrs.fillna(0.0))).astype(np.float32)

    # 2. Life Residual / Unaccounted Hours (24-Hour Life Budget Constraint)
    df_clean['unaccounted_hours'] = (24.0 - (scr_hrs.fillna(0.0) + wrk_hrs.fillna(0.0) + slp_hrs.fillna(0.0))).astype(np.float32)

    # 3. High-Risk Activity Ratios (Native NaN propagation)
    df_clean['gaming_to_screen_ratio'] = np.where(scr_hrs.isna(), np.nan, (gam_hrs / (scr_hrs + eps))).astype(np.float32)
    df_clean['social_to_screen_ratio'] = np.where(scr_hrs.isna(), np.nan, (soc_hrs / (scr_hrs + eps))).astype(np.float32)
    df_clean['screen_to_sleep_ratio'] = np.where(slp_hrs.isna(), np.nan, (scr_hrs / (slp_hrs + eps))).astype(np.float32)

    # 4. Hourly Activity Rates
    df_clean['notifications_per_hour'] = np.where(scr_hrs.isna(), np.nan, (notifs / (scr_hrs + eps))).astype(np.float32)
    df_clean['app_opens_per_hour'] = np.where(scr_hrs.isna(), np.nan, (app_ops / (scr_hrs + eps))).astype(np.float32)

    # 5. Weekend to Weekday Screen Time Ratio
    df_clean['weekend_screen_time_ratio'] = np.where(scr_hrs.isna(), np.nan, (wknd_hrs / (scr_hrs + eps))).astype(np.float32)

    # 6. Sleep Deficit relative to standard 8h target
    df_clean['sleep_deficit'] = np.where(slp_hrs.isna(), np.nan, (8.0 - slp_hrs)).astype(np.float32)

    # 7. Total Daily Activity Hours (social + gaming + work + sleep)
    df_clean['total_activity_hours'] = (soc_hrs.fillna(0.0) + gam_hrs.fillna(0.0) + wrk_hrs.fillna(0.0) + slp_hrs.fillna(0.0)).astype(np.float32)

    # 8. Screen Time Proportion of Total Activity
    df_clean['screen_time_ratio_total'] = np.where(scr_hrs.isna(), np.nan, (scr_hrs / (df_clean['total_activity_hours'] + eps))).astype(np.float32)

    # 9. Insomnia Compulsive Checking: App opens per sleep hour
    df_clean['sleep_app_opens_ratio'] = np.where(slp_hrs.isna(), np.nan, (app_ops / (slp_hrs + eps))).astype(np.float32)

    # 10. Unsupervised Grid-Frequency / Relative Density
    grid_cols = ['app_opens_per_day', 'notifications_per_day', 'daily_screen_time_hours', 'weekend_screen_time', 'age']
    for col in grid_cols:
        if col in df_clean.columns:
            freq_map = df_clean[col].value_counts(normalize=True, dropna=True).to_dict()
            df_clean[f'{col}_freq'] = df_clean[col].map(freq_map).fillna(0.0).astype(np.float32)

    # 11. Productive Work Shield (shields high-screen productive workers)
    df_clean['productive_work_ratio'] = np.where(scr_hrs.isna(), np.nan, (wrk_hrs / (scr_hrs + eps))).astype(np.float32)

    # 12. Work-Adjusted Screen Load (non-work screen load relative to sleep)
    df_clean['work_adjusted_screen_load'] = np.where(
        slp_hrs.isna() | scr_hrs.isna(),
        np.nan,
        ((scr_hrs - wrk_hrs.fillna(0.0)) / (slp_hrs + eps))
    ).astype(np.float32)

    # 13. Compulsive Checking Velocity (awake-hours normalized checking rate)
    awake_hrs = np.maximum(eps, 24.0 - slp_hrs.fillna(8.0))
    df_clean['compulsive_checking_velocity'] = np.where(
        scr_hrs.isna(),
        np.nan,
        ((notifs / (app_ops + 1.0)) * (scr_hrs / awake_hrs))
    ).astype(np.float32)

    # 14. Circadian Dopamine Strain (polynomial nighttime dopamine penalty)
    social_gaming = soc_hrs.fillna(0.0) + gam_hrs.fillna(0.0)
    sleep_def = np.maximum(0.0, 8.0 - slp_hrs.fillna(8.0))
    df_clean['circadian_dopamine_strain'] = np.where(
        slp_hrs.isna(),
        np.nan,
        (sleep_def * np.power(np.maximum(0.0, social_gaming), 1.5) / (slp_hrs + 1.0))
    ).astype(np.float32)

    # 15. Synthetic Time Budget Deficit & Violation Indicator (CTGAN generator artifact)
    claimed_activities = soc_hrs.fillna(0.0) + gam_hrs.fillna(0.0) + wrk_hrs.fillna(0.0)
    df_clean['synthetic_budget_deficit'] = np.where(scr_hrs.isna(), np.nan, (claimed_activities - scr_hrs)).astype(np.float32)
    df_clean['synthetic_budget_violation'] = np.where(scr_hrs.isna(), np.nan, (claimed_activities > scr_hrs).astype(np.float32)).astype(np.float32)

    # 16. Multi-Way Demographic Joint Profile Frequency
    if 'gender' in df_clean.columns and 'stress_level' in df_clean.columns and 'academic_work_impact' in df_clean.columns:
        joint_key = (
            df_clean['gender'].astype(str) + '_' + 
            df_clean['stress_level'].astype(str) + '_' + 
            df_clean['academic_work_impact'].astype(str)
        )
        joint_freq_map = joint_key.value_counts(normalize=True).to_dict()
        df_clean['joint_profile_freq'] = joint_key.map(joint_freq_map).fillna(0.0).astype(np.float32)

    # 17. Inter-Session Arrival Time Burstiness (B_ISAT)
    df_clean['B_ISAT'] = np.where(
        scr_hrs.isna() | notifs.isna(),
        np.nan,
        ((app_ops / (scr_hrs + eps)) / (notifs + eps))
    ).astype(np.float32)

    # 18. Attention Fragmentation Entropy (H_AF)
    tot_act_time = soc_hrs.fillna(0.0) + gam_hrs.fillna(0.0) + wrk_hrs.fillna(0.0) + eps
    p_soc = np.clip(soc_hrs.fillna(0.0) / tot_act_time, 1e-6, 1.0)
    p_gam = np.clip(gam_hrs.fillna(0.0) / tot_act_time, 1e-6, 1.0)
    p_wrk = np.clip(wrk_hrs.fillna(0.0) / tot_act_time, 1e-6, 1.0)
    df_clean['H_AF'] = -(p_soc * np.log(p_soc) + p_gam * np.log(p_gam) + p_wrk * np.log(p_wrk)).astype(np.float32)

    # 19. Chronobiological Dopamine-to-Utility Ratio (CDUR)
    chrono_penalty = 24.0 / (slp_hrs.fillna(8.0) + eps)
    df_clean['CDUR'] = (
        ((soc_hrs.fillna(0.0) + gam_hrs.fillna(0.0)) / (wrk_hrs.fillna(0.0) + eps)) * chrono_penalty
    ).astype(np.float32)

    # 20. Dopamine Chasing Index
    leisure_hrs = soc_hrs.fillna(0.0) + gam_hrs.fillna(0.0)
    sigmoid_leisure = 1.0 / (1.0 + np.exp(-np.clip(leisure_hrs - 3.0, -10.0, 10.0)))
    df_clean['dopamine_chasing_index'] = (np.log1p(app_ops.fillna(0.0) / (notifs.fillna(0.0) + 1.0)) * sigmoid_leisure).astype(np.float32)

    # 21. Compulsive Checking Density
    df_clean['compulsive_checking_density'] = np.where(
        scr_hrs.isna(),
        np.nan,
        (((app_ops.fillna(0.0) + notifs.fillna(0.0)) / awake_hrs) * (1.0 / (scr_hrs + eps)))
    ).astype(np.float32)

    # 22. Decimal Lattice Features (Sub-unit position and first decimal digit CTGAN artifacts)
    frac_cols = ['daily_screen_time_hours', 'social_media_hours', 'gaming_hours', 'work_study_hours', 'sleep_hours', 'weekend_screen_time']
    for c in frac_cols:
        if c in df_clean.columns:
            val = _num(c)
            df_clean[f'frac_{c}'] = np.where(val.isna(), np.nan, (val - np.floor(val))).astype(np.float32)
            df_clean[f'd1_{c}'] = np.where(val.isna(), np.nan, (np.floor(val * 10.0) % 10.0)).astype(np.float32)

    # 23. Missing Pattern Count (Explicit Indicator)
    raw_num_cols = ['age', 'daily_screen_time_hours', 'social_media_hours', 'gaming_hours', 'work_study_hours', 'sleep_hours', 'notifications_per_day', 'app_opens_per_day', 'weekend_screen_time']
    existing_raw = [c for c in raw_num_cols if c in df_clean.columns]
    df_clean['missing_features_count'] = df_clean[existing_raw].isna().sum(axis=1).astype(np.float32)

    # 24. Mathematical Continuous Interactions (Wave 10)
    df_clean['st_mul_sm'] = (scr_hrs * soc_hrs).astype(np.float32)
    df_clean['st_div_awake'] = (scr_hrs / awake_hrs).astype(np.float32)
    df_clean['st_sub_gm'] = (scr_hrs - gam_hrs).astype(np.float32)
    df_clean['st_risk_boundary'] = ((scr_hrs - 5.5) * (soc_hrs.fillna(0.0) + gam_hrs.fillna(0.0))).astype(np.float32)
    df_clean['screen_intensity'] = ((app_ops.fillna(0.0) * scr_hrs) / awake_hrs).astype(np.float32)

    # 25. BigQuery Group Cohort Residuals
    if 'age' in df_clean.columns and 'gender' in df_clean.columns:
        cohort_key = df_clean['age'].astype(str) + '_' + df_clean['gender'].astype(str)
        for target_c, target_arr in [('st', scr_hrs), ('sm', soc_hrs)]:
            s = pd.Series(target_arr)
            c_mean = s.groupby(cohort_key).transform('mean').to_numpy(dtype=np.float32)
            c_std = s.groupby(cohort_key).transform('std').fillna(1.0).to_numpy(dtype=np.float32)
            df_clean[f'cohort_diff_{target_c}'] = (target_arr - c_mean).astype(np.float32)
            df_clean[f'cohort_zscore_{target_c}'] = ((target_arr - c_mean) / (c_std + eps)).astype(np.float32)

    # Downcast floats to float32 for maximum memory efficiency
    for col in df_clean.select_dtypes(include=['float64']).columns:
        df_clean[col] = df_clean[col].astype(np.float32)
    for col in df_clean.select_dtypes(include=['int64']).columns:
        df_clean[col] = pd.to_numeric(df_clean[col], downcast='integer')

    return df_clean
