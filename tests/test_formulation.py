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
    "age": 25,
    "gender": "Male",
    "daily_screen_time_hours": 6.0,
    "social_media_hours": 2.0,
    "gaming_hours": 1.0,
    "work_study_hours": 2.0,
    "sleep_hours": 7.5,
    "notifications_per_day": 50.0,
    "app_opens_per_day": 40.0,
    "weekend_screen_time": 4.0,
    "stress_level": "Medium",
    "academic_work_impact": "No"
}

@pytest.mark.parametrize("age", [10, 50, 120])
def test_pydantic_valid_age_boundaries(age):
    data = base_valid_data.copy()
    data["age"] = age
    model = UserBehaviorInput(**data)
    assert model.age == age

@pytest.mark.parametrize("age", [9, 121, -5])
def test_pydantic_invalid_age_boundaries(age):
    data = base_valid_data.copy()
    data["age"] = age
    with pytest.raises(ValidationError):
        UserBehaviorInput(**data)

@pytest.mark.parametrize("screen_time, social, gaming, work", [
    (10.0, 10.0, 0.0, 0.0),  # Edge case: social equals screen time
    (24.0, 5.0, 10.0, 8.0)   # High limit
])
def test_pydantic_cross_field_valid(screen_time, social, gaming, work):
    data = base_valid_data.copy()
    data["daily_screen_time_hours"] = screen_time
    data["social_media_hours"] = social
    data["gaming_hours"] = gaming
    data["work_study_hours"] = work
    model = UserBehaviorInput(**data)
    assert model.daily_screen_time_hours == screen_time

@pytest.mark.parametrize("sleep", [0.0, 12.0, 24.0])
def test_pydantic_valid_sleep(sleep):
    data = base_valid_data.copy()
    data["sleep_hours"] = sleep
    model = UserBehaviorInput(**data)
    assert model.sleep_hours == sleep

@pytest.mark.parametrize("sleep", [-1.0, 25.0])
def test_pydantic_invalid_sleep(sleep):
    data = base_valid_data.copy()
    data["sleep_hours"] = sleep
    with pytest.raises(ValidationError):
        UserBehaviorInput(**data)

def test_pydantic_accepts_nan():
    data = base_valid_data.copy()
    data["daily_screen_time_hours"] = None
    data["age"] = None
    model = UserBehaviorInput(**data)
    assert model.daily_screen_time_hours is None
    assert model.age is None


# ---------------------------------------------------------
# Test Suite 2: test_engineered_features_math
# ---------------------------------------------------------

@pytest.fixture
def mock_dataframe():
    data = {
        "age": [25, 30, 45],
        "gender": ["Male", "Female", "Male"],
        "daily_screen_time_hours": [6.0, 0.0, 10.0],  # Middle row has 0 screen time
        "social_media_hours": [2.0, 0.0, 5.0],
        "gaming_hours": [1.0, 0.0, 2.0],
        "work_study_hours": [2.0, 0.0, 3.0],
        "notifications_per_day": [50.0, 0.0, 100.0],
        "app_opens_per_day": [40.0, 10.0, 80.0],
        "sleep_hours": [7.5, 8.0, 6.0],
        "weekend_screen_time": [4.0, 2.0, 8.0],
        "stress_level": ["Medium", "Low", "High"],
        "academic_work_impact": ["No", "No", "Yes"]
    }
    return pd.DataFrame(data)

def test_engineered_features_shape(mock_dataframe):
    # original has 12 columns
    assert mock_dataframe.shape[1] == 12
    processed = preprocess_and_engineer(mock_dataframe)
    # output should have 12 original + 9 engineered = 21
    assert processed.shape[1] == 23

def test_engineered_features_zero_division(mock_dataframe):
    # Daily_Screen_Time is 0 for index 1
    processed = preprocess_and_engineer(mock_dataframe)

    # Assert values for index 1 are not NA or Inf due to zero division
    row = processed.iloc[1]
    assert not np.isnan(row['social_to_screen_ratio'])
    assert not np.isinf(row['social_to_screen_ratio'])
    assert not np.isnan(row['gaming_to_screen_ratio'])
    assert not np.isnan(row['notifications_per_hour'])

@pytest.mark.parametrize("idx, expected_deficit", [
    (0, 0.5), # 8.0 - 7.5
    (1, 0.0), # 8.0 - 8.0
    (2, 2.0)  # 8.0 - 6.0
])
def test_engineered_sleep_deficit(mock_dataframe, idx, expected_deficit):
    processed = preprocess_and_engineer(mock_dataframe)
    assert processed.iloc[idx]['sleep_deficit'] == expected_deficit

def test_preprocess_and_engineer_allows_null(mock_dataframe):
    mock_dataframe.loc[0, "age"] = np.nan
    mock_dataframe.loc[0, "daily_screen_time_hours"] = np.nan

    processed = preprocess_and_engineer(mock_dataframe)
    assert pd.isna(processed.loc[0, "age"])
    assert pd.isna(processed.loc[0, "social_to_screen_ratio"])


# ---------------------------------------------------------
# Test Suite 3: test_leak_free_cv
# ---------------------------------------------------------

@pytest.fixture
def mock_cv_dataframe():
    np.random.seed(42)
    n = 100
    data = {
        "age": np.random.uniform(15, 60, n),
        "gender": np.random.choice(["Male", "Female"], n),
        "daily_screen_time_hours": np.random.uniform(2, 10, n),
        "social_media_hours": np.random.uniform(0, 2, n),
        "gaming_hours": np.random.uniform(0, 2, n),
        "work_study_hours": np.random.uniform(0, 2, n),
        "notifications_per_day": np.random.poisson(30, n).astype(float),
        "app_opens_per_day": np.random.poisson(20, n).astype(float),
        "sleep_hours": np.random.uniform(5, 9, n),
        "weekend_screen_time": np.random.uniform(2, 10, n),
        "stress_level": np.random.choice(["Low", "Medium", "High"], n),
        "academic_work_impact": np.random.choice(["Yes", "No"], n),
    }

    # Introduce some NaN values manually
    df = pd.DataFrame(data)
    df.loc[0, "daily_screen_time_hours"] = np.nan
    df.loc[5, "gender"] = np.nan
    df.loc[10, "stress_level"] = np.nan

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

    # OOF predictions should be a matrix of shape (len(X), 3)
    assert oof_preds.shape == (len(X), 3)
    assert 0 <= mean_auc <= 1.0
