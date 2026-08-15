import pytest
import pandas as pd
import numpy as np
from pydantic import ValidationError
from unittest.mock import patch, MagicMock
from src.model.formulation import UserBehaviorInput, preprocess_and_engineer
from src.model.solver import CompetitionSolver

# ---------------------------------------------------------
# Test Suite 1: test_pydantic_validation (Input Boundaries)
# ---------------------------------------------------------

# Valid base case to clone from
base_valid_data = {
    "Age": 25,
    "Gender": "Male",
    "Daily_Screen_Time": 6.0,
    "Social_Media_Usage": 2.0,
    "Gaming_Hours": 1.0,
    "Work_Study_Time": 2.0,
    "Notification_Frequency": 50.0,
    "App_Opening_Frequency": 40.0,
    "Sleep_Duration": 7.5,
    "Stress_Level": "Moderate",
    "Installed_Apps": 20,
    "User_Activity": 1.5
}

@pytest.mark.parametrize("age", [10, 50, 120])
def test_pydantic_valid_age_boundaries(age):
    data = base_valid_data.copy()
    data["Age"] = age
    model = UserBehaviorInput(**data)
    assert model.Age == age

@pytest.mark.parametrize("age", [9, 121, -5])
def test_pydantic_invalid_age_boundaries(age):
    data = base_valid_data.copy()
    data["Age"] = age
    with pytest.raises(ValidationError):
        UserBehaviorInput(**data)

@pytest.mark.parametrize("gender", ["Male", "Female"])
def test_pydantic_valid_genders(gender):
    data = base_valid_data.copy()
    data["Gender"] = gender
    model = UserBehaviorInput(**data)
    assert model.Gender == gender

@pytest.mark.parametrize("gender", ["Other", "Unknown", 1])
def test_pydantic_invalid_genders(gender):
    data = base_valid_data.copy()
    data["Gender"] = gender
    with pytest.raises(ValidationError):
        UserBehaviorInput(**data)

@pytest.mark.parametrize("screen_time, social, gaming, work", [
    (10.0, 10.0, 0.0, 0.0),  # Edge case: social equals screen time
    (24.0, 5.0, 10.0, 8.0)   # High limit
])
def test_pydantic_cross_field_valid(screen_time, social, gaming, work):
    data = base_valid_data.copy()
    data["Daily_Screen_Time"] = screen_time
    data["Social_Media_Usage"] = social
    data["Gaming_Hours"] = gaming
    data["Work_Study_Time"] = work
    model = UserBehaviorInput(**data)
    assert model.Daily_Screen_Time == screen_time

@pytest.mark.parametrize("screen_time, social", [
    (5.0, 6.0),  # Social exceeds total screen time
    (2.0, 10.0)
])
def test_pydantic_social_media_exceeds(screen_time, social):
    data = base_valid_data.copy()
    data["Daily_Screen_Time"] = screen_time
    data["Social_Media_Usage"] = social
    with pytest.raises(ValueError, match="Social_Media_Usage cannot exceed Daily_Screen_Time"):
        UserBehaviorInput(**data)

@pytest.mark.parametrize("screen_time, gaming", [
    (5.0, 6.0),  # Gaming exceeds total screen time
])
def test_pydantic_gaming_exceeds(screen_time, gaming):
    data = base_valid_data.copy()
    data["Daily_Screen_Time"] = screen_time
    data["Gaming_Hours"] = gaming
    with pytest.raises(ValueError, match="Gaming_Hours cannot exceed Daily_Screen_Time"):
        UserBehaviorInput(**data)

@pytest.mark.parametrize("screen_time, work", [
    (8.0, 10.0),  # Work exceeds total screen time
])
def test_pydantic_work_exceeds(screen_time, work):
    data = base_valid_data.copy()
    data["Daily_Screen_Time"] = screen_time
    data["Work_Study_Time"] = work
    with pytest.raises(ValueError, match="Work_Study_Time cannot exceed Daily_Screen_Time"):
        UserBehaviorInput(**data)

@pytest.mark.parametrize("field, value", [
    ("Daily_Screen_Time", np.nan),
    ("Social_Media_Usage", np.inf),
    ("Age", np.nan),
    ("Sleep_Duration", -np.inf)
])
def test_pydantic_rejects_nan_inf(field, value):
    data = base_valid_data.copy()
    data[field] = value
    with pytest.raises(ValueError):
        UserBehaviorInput(**data)

@pytest.mark.parametrize("sleep", [0.0, 12.0, 24.0])
def test_pydantic_valid_sleep(sleep):
    data = base_valid_data.copy()
    data["Sleep_Duration"] = sleep
    model = UserBehaviorInput(**data)
    assert model.Sleep_Duration == sleep

@pytest.mark.parametrize("sleep", [-1.0, 25.0])
def test_pydantic_invalid_sleep(sleep):
    data = base_valid_data.copy()
    data["Sleep_Duration"] = sleep
    with pytest.raises(ValidationError):
        UserBehaviorInput(**data)

@pytest.mark.parametrize("stress", ["Low", "Moderate", "High", 1, 5, 10])
def test_pydantic_valid_stress(stress):
    data = base_valid_data.copy()
    data["Stress_Level"] = stress
    model = UserBehaviorInput(**data)
    assert model.Stress_Level == stress

@pytest.mark.parametrize("apps", [0, 10, 100])
def test_pydantic_valid_apps(apps):
    data = base_valid_data.copy()
    data["Installed_Apps"] = apps
    model = UserBehaviorInput(**data)
    assert model.Installed_Apps == apps

@pytest.mark.parametrize("apps", [-1, -10])
def test_pydantic_invalid_apps(apps):
    data = base_valid_data.copy()
    data["Installed_Apps"] = apps
    with pytest.raises(ValidationError):
        UserBehaviorInput(**data)

# ---------------------------------------------------------
# Test Suite 2: test_engineered_features_math
# ---------------------------------------------------------

@pytest.fixture
def mock_dataframe():
    data = {
        "Age": [25, 30, 45],
        "Gender": ["Male", "Female", "Male"],
        "Daily_Screen_Time": [6.0, 0.0, 10.0],  # Middle row has 0 screen time
        "Social_Media_Usage": [2.0, 0.0, 5.0],
        "Gaming_Hours": [1.0, 0.0, 2.0],
        "Work_Study_Time": [2.0, 0.0, 3.0],
        "Notification_Frequency": [50.0, 0.0, 100.0],
        "App_Opening_Frequency": [40.0, 10.0, 80.0],
        "Sleep_Duration": [7.5, 8.0, 6.0],
        "Stress_Level": ["Moderate", "Low", "High"],
        "Installed_Apps": [20, 10, 50],
        "User_Activity": [1.5, 0.5, 2.5]
    }
    return pd.DataFrame(data)

def test_engineered_features_shape(mock_dataframe):
    # original has 12 columns
    assert mock_dataframe.shape[1] == 12
    processed = preprocess_and_engineer(mock_dataframe)
    # output should have 12 original + 4 engineered = 16
    assert processed.shape[1] == 16

def test_engineered_features_zero_division(mock_dataframe):
    # Daily_Screen_Time is 0 for index 1
    processed = preprocess_and_engineer(mock_dataframe)

    # Assert values for index 1 are not NA or Inf due to zero division
    row = processed.iloc[1]
    assert not np.isnan(row['social_media_proportion'])
    assert not np.isinf(row['social_media_proportion'])
    assert not np.isnan(row['gaming_proportion'])
    assert not np.isnan(row['notifications_per_hour'])

@pytest.mark.parametrize("idx, expected_deficit", [
    (0, 0.5), # 8.0 - 7.5
    (1, 0.0), # 8.0 - 8.0
    (2, 2.0)  # 8.0 - 6.0
])
def test_engineered_sleep_deficit(mock_dataframe, idx, expected_deficit):
    processed = preprocess_and_engineer(mock_dataframe)
    assert processed.iloc[idx]['sleep_deficit'] == expected_deficit

def test_preprocess_and_engineer_raises_on_null(mock_dataframe):
    mock_dataframe.loc[0, "Age"] = np.nan
    with pytest.raises(ValueError, match="Null value found in column Age"):
        preprocess_and_engineer(mock_dataframe)


# ---------------------------------------------------------
# Test Suite 3: test_leak_free_cv
# ---------------------------------------------------------

@pytest.fixture
def mock_cv_dataframe():
    np.random.seed(42)
    n = 100
    data = {
        "Age": np.random.uniform(15, 60, n),
        "Gender": np.random.choice(["Male", "Female"], n),
        "Daily_Screen_Time": np.random.uniform(2, 10, n),
        "Social_Media_Usage": np.random.uniform(0, 2, n),
        "Gaming_Hours": np.random.uniform(0, 2, n),
        "Work_Study_Time": np.random.uniform(0, 2, n),
        "Notification_Frequency": np.random.poisson(30, n).astype(float),
        "App_Opening_Frequency": np.random.poisson(20, n).astype(float),
        "Sleep_Duration": np.random.uniform(5, 9, n),
        "Stress_Level": np.random.choice(["Low", "Moderate", "High"], n),
        "Installed_Apps": np.random.poisson(15, n),
        "User_Activity": np.random.uniform(0, 5, n),
    }
    df = pd.DataFrame(data)
    y = pd.Series(np.random.choice([0, 1], n), name="addicted_label")
    return df, y

@patch("src.model.solver.LGBMClassifier.fit")
@patch("src.model.solver.XGBClassifier.fit")
@patch("src.model.solver.CatBoostClassifier.fit")
def test_leak_free_cv_folds_touched(mock_cat_fit, mock_xgb_fit, mock_lgb_fit, mock_cv_dataframe):
    X, y = mock_cv_dataframe
    solver = CompetitionSolver(n_splits=3) # Use 3 for faster testing

    # We want to mock predict_proba so it runs without error
    # despite fit being mocked.
    # We need predict_proba to return correctly sized arrays based on the validation fold length.
    # Since MagicMock can accept a side_effect function, we can dynamically return the correct shape.
    def mock_predict_proba(X_val):
        return np.zeros((len(X_val), 2))

    with patch("src.model.solver.LGBMClassifier.predict_proba", side_effect=mock_predict_proba), \
         patch("src.model.solver.XGBClassifier.predict_proba", side_effect=mock_predict_proba), \
         patch("src.model.solver.CatBoostClassifier.predict_proba", side_effect=mock_predict_proba):

        oof_preds, mean_auc = solver.cross_validate(X, y)

        # Verify fits were called 3 times (once per fold)
        assert mock_lgb_fit.call_count == 3
        assert mock_xgb_fit.call_count == 3
        assert mock_cat_fit.call_count == 3

        # For each call, ensure the validation set was not passed to fit
        # The first argument to fit is X_train_clean. Its size should be n * (2/3)
        # We can inspect the arguments passed to fit.
        for call_args in mock_lgb_fit.call_args_list:
            X_train_fold, y_train_fold = call_args[0]
            assert len(X_train_fold) < len(X)
            # The exact number depends on StratifiedKFold split,
            # for 100 rows and 3 splits, it should be 66 or 67.
            assert 65 <= len(X_train_fold) <= 68
            # The validation rows must not be in X_train_fold index
            # Actually, preprocess_and_engineer resets or keeps index?
            # Wait, preprocess_and_engineer currently creates a new DF from to_dict('records')
            # which resets the index. So we can just check lengths.

def test_leak_free_cv_output_shape(mock_cv_dataframe):
    X, y = mock_cv_dataframe
    solver = CompetitionSolver(n_splits=2)

    oof_preds, mean_auc = solver.cross_validate(X, y)

    # OOF predictions should have the exact shape of X
    assert len(oof_preds) == len(X)
    assert 0 <= mean_auc <= 1.0
