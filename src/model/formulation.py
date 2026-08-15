from typing import List, Literal, Optional, Union
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, model_validator, ConfigDict

class UserBehaviorInput(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    Age: float = Field(..., ge=10, le=120)
    Gender: Literal['Male', 'Female']
    Daily_Screen_Time: float = Field(..., ge=0.0, le=24.0)
    Social_Media_Usage: float = Field(..., ge=0.0, le=24.0)
    Gaming_Hours: float = Field(..., ge=0.0, le=24.0)
    Work_Study_Time: float = Field(..., ge=0.0, le=24.0)
    Notification_Frequency: float = Field(..., ge=0.0)
    App_Opening_Frequency: float = Field(..., ge=0.0)
    Sleep_Duration: float = Field(..., ge=0.0, le=24.0)
    Stress_Level: Union[int, Literal['Low', 'Moderate', 'High']]
    Installed_Apps: int = Field(..., ge=0)
    User_Activity: float = Field(..., ge=0.0)

    @model_validator(mode='after')
    def check_sub_durations(self):
        if self.Social_Media_Usage > self.Daily_Screen_Time:
            raise ValueError("Social_Media_Usage cannot exceed Daily_Screen_Time")
        if self.Gaming_Hours > self.Daily_Screen_Time:
            raise ValueError("Gaming_Hours cannot exceed Daily_Screen_Time")
        if self.Work_Study_Time > self.Daily_Screen_Time:
            raise ValueError("Work_Study_Time cannot exceed Daily_Screen_Time")

        # Check for NaN or Inf (float values)
        for field in ['Age', 'Daily_Screen_Time', 'Social_Media_Usage', 'Gaming_Hours',
                      'Work_Study_Time', 'Notification_Frequency', 'App_Opening_Frequency',
                      'Sleep_Duration', 'User_Activity']:
            val = getattr(self, field)
            if np.isnan(val) or np.isinf(val):
                raise ValueError(f"{field} cannot be NaN or Inf")

        return self

def preprocess_and_engineer(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validates input and engineers 4 advanced behavioral features:
    1) social_media_proportion
    2) gaming_proportion
    3) notifications_per_hour
    4) sleep_deficit
    """
    # Defensive copy to avoid mutating the original
    df = df.copy()

    # 1. Pydantic Validation
    # We do a row-by-row validation or validate dicts
    records = df.to_dict(orient='records')
    validated_records = []
    for r in records:
        # Convert any potential string nan to float nan
        for k, v in r.items():
            if pd.isna(v):
                raise ValueError(f"Null value found in column {k}")

        validated = UserBehaviorInput(**r)
        validated_records.append(validated.model_dump())

    df_clean = pd.DataFrame(validated_records)

    # 2. Feature Engineering
    epsilon = 1e-6

    # a) social_media_proportion: Social media usage as a proportion of total screen time
    df_clean['social_media_proportion'] = df_clean['Social_Media_Usage'] / (df_clean['Daily_Screen_Time'] + epsilon)

    # b) gaming_proportion: Gaming hours relative to total screen time
    df_clean['gaming_proportion'] = df_clean['Gaming_Hours'] / (df_clean['Daily_Screen_Time'] + epsilon)

    # c) notifications_per_hour: Notifications divided by (total screen time + epsilon)
    df_clean['notifications_per_hour'] = df_clean['Notification_Frequency'] / (df_clean['Daily_Screen_Time'] + epsilon)

    # d) sleep_deficit: Target sleep hours (e.g., 8) minus actual sleep hours
    df_clean['sleep_deficit'] = 8.0 - df_clean['Sleep_Duration']

    # The output should have 12 original features + 4 engineered = 16 columns
    return df_clean
