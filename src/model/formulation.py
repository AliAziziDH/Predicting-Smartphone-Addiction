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
    Clean, Vectorized, Leak-Free Feature Engineering Pipeline.
    Strictly preserves native NaN propagation for optimal GBDT tree splits.
    Focuses exclusively on stable, high-generalization domain ratios and time balances.
    """
    df_clean = df.copy()
    eps = 1e-5

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

    # 1. Residual Screen Time (Other Screen)
    df_clean['other_screen'] = (scr_hrs - (soc_hrs.fillna(0.0) + gam_hrs.fillna(0.0) + wrk_hrs.fillna(0.0))).astype(np.float32)

    # 2. 24-Hour Life Budget Residual
    df_clean['unaccounted_hours'] = (24.0 - (scr_hrs + wrk_hrs + slp_hrs)).astype(np.float32)

    # 2.5 Unaccounted Time Leakage (UTL)
    df_clean['UTL'] = (scr_hrs - (soc_hrs + gam_hrs + wrk_hrs)).astype(np.float32)
    df_clean['UTL_ratio'] = (df_clean['UTL'] / (scr_hrs + eps)).astype(np.float32)

    # 3. High-Risk Activity Ratios
    df_clean['gaming_to_screen_ratio'] = np.where(scr_hrs.isna(), np.nan, (gam_hrs / (scr_hrs + eps))).astype(np.float32)
    df_clean['social_to_screen_ratio'] = np.where(scr_hrs.isna(), np.nan, (soc_hrs / (scr_hrs + eps))).astype(np.float32)
    df_clean['screen_to_sleep_ratio'] = np.where(slp_hrs.isna(), np.nan, (scr_hrs / (slp_hrs + eps))).astype(np.float32)

    # 4. Hourly Rates & Checking Intensity
    df_clean['notifications_per_hour'] = np.where(scr_hrs.isna(), np.nan, (notifs / (scr_hrs + eps))).astype(np.float32)
    df_clean['app_opens_per_hour'] = np.where(scr_hrs.isna(), np.nan, (app_ops / (scr_hrs + eps))).astype(np.float32)
    df_clean['compulsive_pull_ratio'] = (app_ops / (notifs + 1.0)).astype(np.float32)

    # 5. Weekend / Work / Sleep Balance
    df_clean['weekend_screen_time_ratio'] = np.where(scr_hrs.isna(), np.nan, (wknd_hrs / (scr_hrs + eps))).astype(np.float32)
    df_clean['sleep_deficit'] = np.where(slp_hrs.isna(), np.nan, (8.0 - slp_hrs)).astype(np.float32)
    df_clean['productive_work_ratio'] = np.where(scr_hrs.isna(), np.nan, (wrk_hrs / (scr_hrs + eps))).astype(np.float32)
    df_clean['work_adjusted_screen_load'] = np.where(
        slp_hrs.isna() | scr_hrs.isna(),
        np.nan,
        ((scr_hrs - wrk_hrs.fillna(0.0)) / (slp_hrs + eps))
    ).astype(np.float32)

    # 5.5 Work Shield Factor
    leisure_hours = soc_hrs + gam_hrs
    df_clean['work_shield_factor'] = ((wrk_hrs / (scr_hrs + eps)) * np.exp(-leisure_hours / 2.0)).astype(np.float32)

    # 6. Missingness Indicator
    raw_cols = ['age', 'daily_screen_time_hours', 'social_media_hours', 'gaming_hours', 'work_study_hours', 'sleep_hours', 'notifications_per_day', 'app_opens_per_day', 'weekend_screen_time']
    existing_raw = [c for c in raw_cols if c in df_clean.columns]
    df_clean['missing_features_count'] = df_clean[existing_raw].isna().sum(axis=1).astype(np.float32)

    # Downcast floats to float32 for maximum memory efficiency
    for col in df_clean.select_dtypes(include=['float64']).columns:
        df_clean[col] = df_clean[col].astype(np.float32)
    for col in df_clean.select_dtypes(include=['int64']).columns:
        df_clean[col] = pd.to_numeric(df_clean[col], downcast='integer')

    return df_clean
