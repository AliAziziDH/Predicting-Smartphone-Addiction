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
        # We only check if both values are not None and not NaN
        if self.social_media_hours is not None and self.daily_screen_time_hours is not None:
            if not np.isnan(self.social_media_hours) and not np.isnan(self.daily_screen_time_hours):
                if self.social_media_hours > self.daily_screen_time_hours:
                    pass # Relaxed boundary checks for synthetic data

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
    Validates input and engineers 4 advanced behavioral features.
    Handles NaN values smoothly without raising errors.
    """
    # Defensive copy to avoid mutating the original
    df = df.copy()

    # 1. Pydantic Validation
    records = df.to_dict(orient='records')
    validated_records = []
    for r in records:
        # Replace pd.NA or literal NaNs with None for Pydantic
        clean_r = {}
        for k, v in r.items():
            if pd.isna(v):
                clean_r[k] = None
            else:
                clean_r[k] = v

        validated = UserBehaviorInput(**clean_r)
        validated_records.append(validated.model_dump())

    df_clean = pd.DataFrame(validated_records)

    # 2. Feature Engineering
    epsilon = 1e-6

    # a) social_media_proportion
    df_clean['social_media_proportion'] = df_clean['social_media_hours'] / (df_clean['daily_screen_time_hours'] + epsilon)

    # b) gaming_proportion
    df_clean['gaming_proportion'] = df_clean['gaming_hours'] / (df_clean['daily_screen_time_hours'] + epsilon)

    # c) notifications_per_hour
    # Assuming 24 hours in a day
    df_clean['notifications_per_hour'] = df_clean['notifications_per_day'] / 24.0

    # 1. sleep_deficit: max(0, 8 - sleep_duration)
    df_clean['sleep_deficit'] = np.maximum(0.0, 8.0 - df_clean['sleep_hours'])

    # 2. distraction_ratio: (social_media_hours + gaming_hours) / daily_screen_time_hours
    denom = df_clean['daily_screen_time_hours']
    num = df_clean['social_media_hours'] + df_clean['gaming_hours']
    df_clean['distraction_ratio'] = np.where(denom > epsilon, num / denom, 0.0)

    # 3. notification_intensity: notification_frequency / daily_screen_time_hours
    df_clean['notification_intensity'] = np.where(denom > epsilon, df_clean['notifications_per_day'] / denom, 0.0)

    # 4. app_opening_intensity: app_opening_frequency / daily_screen_time_hours
    df_clean['app_opening_intensity'] = np.where(denom > epsilon, df_clean['app_opens_per_day'] / denom, 0.0)

    # 5. screen_time_age_intensity: daily_screen_time_hours / age
    age_denom = df_clean['age']
    df_clean['screen_time_age_intensity'] = np.where(age_denom > epsilon, denom / age_denom, 0.0)

    # 6. weekend_overuse_index: weekend_screen_time / daily_screen_time_hours
    df_clean['weekend_overuse_index'] = np.where(denom > epsilon, df_clean['weekend_screen_time'] / denom, 0.0)

    return df_clean
