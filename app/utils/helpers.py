# app/utils/helpers.py


import sys
import os
from typing import Dict, Any, Optional
from pathlib import Path
import pandas as pd
import numpy as np

# Ensure project root is on path
ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from src import get_path, load_model  # noqa: E402
from app.components.feature_forecast_builder import FeatureForecastBuilder  # noqa: E402




MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {
    # WEEK 2 – Classical Time Series

    # "arima_514": {
    #     "week": 2,
    #     "model_type": "arima",
    #     "filename": "arima_514.pkl",
    #     "label": "ARIMA (5,1,4)"
    # },

    # SARIMA
  
    # "sarima_300_0117": {
    #     "week": 2,
    #     "model_type": "sarima",
    #     "filename": "sarima_300_0117.pkl",
    #     "label": "SARIMA (3,0,0)x(0,1,1,7)"
    # },
  

    # ETS
    # "ets_week2": {
    #     "week": 2,
    #     "model_type": "ets",
    #     "filename": "ets_model_week2.pkl",
    #     "label": "ETS (Week 2)"
    # },

    # Prophet
    "prophet_week2": {
        "week": 2,
        "model_type": "prophet",
        "filename": "prophet_model.pkl",
        "label": "Prophet (Week 2)"
    },

    # WEEK 3 – ML Models
    "linear_base": {
        "week": 3,
        "model_type": "linear",
        "filename": "linear_regression.pkl",
        "label": "Linear Regression"
    },
    "linear_random": {
        "week": 3,
        "model_type": "linear",
        "filename": "linear_regression_random.pkl",
        "label": "Linear Regression (Random Search)"
    },
    "linear_grid": {
        "week": 3,
        "model_type": "linear",
        "filename": "linear_regression_grid.pkl",
        "label": "Linear Regression (Grid Search)"
    },
    "linear_hyperopt": {
        "week": 3,
        "model_type": "linear",
        "filename": "linear_regression_hyperopt.pkl",
        "label": "Linear Regression (Hyperopt)"
    },

    # Random Forest
    "rf_base": {
        "week": 3,
        "model_type": "random_forest",
        "filename": "random_forest.pkl",
        "label": "Random Forest"
    },
    "rf_random": {
        "week": 3,
        "model_type": "random_forest",
        "filename": "random_forest_random.pkl",
        "label": "Random Forest (Random Search)"
    },
    "rf_grid": {
        "week": 3,
        "model_type": "random_forest",
        "filename": "random_forest_grid.pkl",
        "label": "Random Forest (Grid Search)"
    },
    "rf_hyperopt": {
        "week": 3,
        "model_type": "random_forest",
        "filename": "random_forest_hyperopt.pkl",
        "label": "Random Forest (Hyperopt)"
    },

    # SVR
    "svr_base": {
        "week": 3,
        "model_type": "svr",
        "filename": "svr.pkl",
        "label": "SVR"
    },
    "svr_random": {
        "week": 3,
        "model_type": "svr",
        "filename": "svr_random.pkl",
        "label": "SVR (Random Search)"
    },
    "svr_grid": {
        "week": 3,
        "model_type": "svr",
        "filename": "svr_grid.pkl",
        "label": "SVR (Grid Search)"
    },
    "svr_hyperopt": {
        "week": 3,
        "model_type": "svr",
        "filename": "svr_hyperopt.pkl",
        "label": "SVR (Hyperopt)"
    },

    # XGBoost
    "xgb_base": {
        "week": 3,
        "model_type": "xgboost",
        "filename": "xgboost.pkl",
        "label": "XGBoost"
    },
    "xgb_random": {
        "week": 3,
        "model_type": "xgboost",
        "filename": "xgboost_random.pkl",
        "label": "XGBoost (Random Search)"
    },
    "xgb_grid": {
        "week": 3,
        "model_type": "xgboost",
        "filename": "xgboost_grid.pkl",
        "label": "XGBoost (Grid Search)"
    },
    "xgb_hyperopt": {
        "week": 3,
        "model_type": "xgboost",
        "filename": "xgboost_hyperopt.pkl",
        "label": "XGBoost (Hyperopt)"
    },

    # LSTM (Keras)
    # "lstm_univariate": {
    #     "week": 3,
    #     "model_type": "lstm",
    #     "filename": "lstm_univariate_model.keras",
    #     "label": "LSTM Univariate"
    # },
    # "lstm_multivariate": {
    #     "week": 3,
    #     "model_type": "lstm",
    #     "filename": "lstm_multivariate_model.keras",
    #     "label": "LSTM Multivariate"
    # },
}


def load_registered_model(model_key: str) -> Optional[Any]:
    """Load a model using the registry metadata."""
    if model_key not in MODEL_REGISTRY:
        print(f"❌ Unknown model key: {model_key}")
        return None

    meta = MODEL_REGISTRY[model_key]
    week = meta["week"]
    model_type = meta["model_type"]
    filename = meta["filename"]

    try:
        if model_type == "lstm":
            from tensorflow import keras  # type: ignore
            model_dir = get_path("lstm_model", week=week)
            load_path = os.path.join(model_dir, filename)
            if not os.path.exists(load_path):
                print(f"❌ LSTM model file not found: {load_path}")
                return None
            print(f"🔄 Loading LSTM model from: {load_path}")
            return keras.models.load_model(load_path)

        # Load other model types
        model = load_model(model_type=model_type, week=week, filename=filename)
        if model is None:
            print(f"❌ Failed to load model: {model_key}")
        return model
        
    except Exception as e:
        print(f"❌ Error loading model {model_key}: {e}")
        return None


# Update FEATURES list
FEATURES = [
    "onpromotion",
    "onpromotion_lag_1",
    "unit_sales_lag_1",
    "unit_sales_lag_3",
    "unit_sales_lag_7",
    "unit_sales_lag_14",
    "unit_sales_lag_30",
    "unit_sales_lag_365",
    "unit_sales_r3_mean",
    "unit_sales_r7_mean",
    "unit_sales_r14_mean",
    "unit_sales_r30_mean",
    "unit_sales_r365_mean",
    "unit_sales_r3_median",
    "unit_sales_r7_median",
    "unit_sales_r14_median",
    "unit_sales_r30_median",
    "unit_sales_r365_median",
    "unit_sales_r7_std",
    "day_of_week",
    "month",
    "quarter",
    "year",
    "days_until_holiday",
    "days_since_holiday",
    "store_avg_sales",
    "store_item_median",
    "month_sin",
    "month_cos",
    "day_of_week_sin",
    "day_of_week_cos",
    "year_sin",
    "year_cos"
]


# Update the build_future_features function to use the builder
def build_future_features(historical, start_date, horizon, onpromotion, frequency="D", model_type="ml"):
    """
    Build future feature matrix for forecasting.
    
    Args:
        historical: Historical DataFrame
        start_date: Start date for forecast
        horizon: Number of periods to forecast
        onpromotion: Promotion status
        frequency: Frequency ('D', 'W', 'M', 'Q', 'Y')
        model_type: Type of model
        
    Returns:
        Future feature DataFrame
    """
    builder = FeatureForecastBuilder(historical)
    
    future_df = builder.build_future_features(
        start_date=pd.to_datetime(start_date),
        horizon=horizon,
        frequency=frequency,
        onpromotion=onpromotion,
        model_type=model_type
    )
    
    return future_df


# Add helper function for creating visualizer
def create_forecast_visualizer(historical_df: pd.DataFrame):
    """
    Create a forecast visualizer with historical data.
    
    Args:
        historical_df: Historical data DataFrame
        
    Returns:
        ForecastVisualizer instance
    """
    from .visualizer import ForecastVisualizer 
    return ForecastVisualizer(historical_df)


# Add helper function for getting model predictions with fallback
def get_model_predictions(model, model_type: str, future_df: pd.DataFrame, 
                         horizon: int, historical_df: pd.DataFrame = None) -> np.ndarray:
    """
    Get predictions from a model with fallback options.
    
    Args:
        model: Trained model
        model_type: Type of model
        future_df: Future features DataFrame
        horizon: Forecast horizon
        historical_df: Historical data for fallback
        
    Returns:
        Array of predictions
    """
    try:
        # Import ForecastEngine
        from ..ui.forecast_engine import ForecastEngine
        from ..components.feature_forecast_builder import FeatureForecastBuilder
        
        # Create builder if historical data is provided
        if historical_df is not None:
            builder = FeatureForecastBuilder(historical_df)
        else:
            builder = None
        
        engine = ForecastEngine(model, model_type, builder)
        return engine.predict(future_df, horizon)
        
    except Exception as e:
        print(f"❌ Error getting predictions: {e}")
        
        # Fallback to historical mean
        if historical_df is not None and 'unit_sales' in historical_df.columns:
            mean_val = historical_df['unit_sales'].mean()
            return np.full(horizon, mean_val)
        else:
            return np.zeros(horizon)