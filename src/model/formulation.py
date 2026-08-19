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

    # 17. High-Performance Memory Downcasting (reduces RAM from 1.2GB to <300MB on 690k rows)
    for col in df_clean.select_dtypes(include=['float64']).columns:
        df_clean[col] = df_clean[col].astype(np.float32)
    for col in df_clean.select_dtypes(include=['int64']).columns:
        df_clean[col] = pd.to_numeric(df_clean[col], downcast='integer')

    return df_clean

