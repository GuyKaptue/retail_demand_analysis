# app/ui/forecast_engine.py - Refactored version

from app.bootstrap import *  # ensures project root is on sys.path  # noqa: F403

import streamlit as st  # type: ignore
import pandas as pd
import numpy as np
from typing import Optional, Tuple

from app.components.feature_forecast_builder import FeatureForecastBuilder

# ---------------------------------------------------------
# Forecasting Engine
# ---------------------------------------------------------


class ForecastEngine:
    """Unified prediction engine for all model types."""
    
    def __init__(self, model, model_type: str, builder: FeatureForecastBuilder):
        """
        Initialize the forecast engine.
        
        Args:
            model: Trained model object
            model_type: Type of model (arima, sarima, ets, prophet, lstm, ml)
            builder: FeatureForecastBuilder instance
        """
        self.model = model
        self.model_type = model_type
        self.builder = builder
    
    def predict(self, future_df: pd.DataFrame, horizon: int, frequency: str = "D") -> np.ndarray:
        """
        Generate predictions using the appropriate method for each model type.
        
        Args:
            future_df: DataFrame with future features
            horizon: Number of periods to forecast
            frequency: Forecast frequency
            
        Returns:
            Array of predictions
        """
        try:
            if self.model_type in ["arima", "sarima"]:
                return self._predict_arima(horizon, frequency)
            elif self.model_type == "ets":
                return self._predict_ets(horizon)
            elif self.model_type == "prophet":
                return self._predict_prophet(future_df)
            elif self.model_type == "lstm":
                return self._predict_lstm(future_df, horizon)
            else:
                return self._predict_ml(future_df)
        except Exception as e:
            st.error(f"Prediction error for {self.model_type}: {str(e)}")
            return self._fallback_predictions(horizon)
    
    def _fallback_predictions(self, horizon: int) -> np.ndarray:
        """Generate fallback predictions using historical mean."""
        st.warning("⚠️ Using historical mean as fallback prediction")
        if len(self.builder.historical_df) > 0 and 'unit_sales' in self.builder.historical_df.columns:
            mean_val = self.builder.historical_df['unit_sales'].mean()
            return np.full(horizon, mean_val)
        else:
            return np.zeros(horizon)
    
    def _predict_arima(self, horizon: int, frequency: str) -> np.ndarray:
        """Predict using ARIMA/SARIMA models with multiple fallback methods."""
        # Method 1: Try custom forecast_future method
        if hasattr(self.model, 'forecast_future'):
            try:
                forecast_result = self.model.forecast_future(
                    forecast_horizon=horizon,
                    frequency=frequency
                )
                if isinstance(forecast_result, dict) and "point_forecasts" in forecast_result:
                    return np.array(forecast_result["point_forecasts"])
                return np.array(forecast_result)
            except Exception as e:
                st.warning(f"Custom forecast method failed: {e}")
        
        # Method 2: Try standard forecast method
        if hasattr(self.model, 'forecast'):
            try:
                result = self.model.forecast(steps=horizon)
                if hasattr(result, 'predicted_mean'):
                    return result.predicted_mean.values
                elif hasattr(result, 'values'):
                    return result.values
                return np.array(result)
            except Exception as e:
                st.warning(f"Standard forecast failed: {e}")
        
        # Method 3: Try get_forecast method (statsmodels)
        if hasattr(self.model, 'get_forecast'):
            try:
                forecast_result = self.model.get_forecast(steps=horizon)
                return forecast_result.predicted_mean.values
            except Exception as e:
                st.warning(f"get_forecast failed: {e}")
        
        # Method 4: Try predict with n parameter (darts models)
        if hasattr(self.model, 'predict'):
            try:
                model_type_name = type(self.model).__name__
                if 'TransferableFutureCovariatesLocalForecastingModel' in model_type_name:
                    return self.model.predict(n=horizon)
                else:
                    start_idx = len(self.builder.historical_df)
                    end_idx = start_idx + horizon - 1
                    return np.array(self.model.predict(start=start_idx, end=end_idx))
            except Exception as e:
                st.warning(f"predict method failed: {e}")
        
        # If all methods failed
        raise ValueError(f"No suitable prediction method found for {self.model_type} model")
    
    def _predict_ets(self, horizon: int) -> np.ndarray:
        """Predict using ETS models."""
        if hasattr(self.model, 'forecast'):
            result = self.model.forecast(steps=horizon)
            if hasattr(result, 'values'):
                return result.values
            return np.array(result)
        else:
            raise ValueError("ETS model does not have forecast method")
    
    def _predict_prophet(self, future_df: pd.DataFrame) -> np.ndarray:
        """Predict using Prophet models."""
        prophet_df = future_df.copy()
        if "date" in prophet_df.columns:
            prophet_df = prophet_df.rename(columns={"date": "ds"})
        
        forecast = self.model.predict(prophet_df)
        return forecast["yhat"].values
    
    def _predict_lstm(self, future_df: pd.DataFrame, horizon: int) -> np.ndarray:
        """Predict using LSTM models."""
        try:
            from tensorflow import keras  # type: ignore  # noqa: F401
        except ImportError:
            raise ImportError("TensorFlow is required for LSTM predictions")
        
        # Check if multivariate
        is_multivariate = "multivariate" in str(type(self.model)).lower() or \
                         (hasattr(self.model, 'input_shape') and 
                          len(self.model.input_shape) > 2 and 
                          self.model.input_shape[2] > 1)
        
        if is_multivariate:
            sequences = self.builder.build_future_sequence_for_lstm(future_df)
            predictions = self.model.predict(sequences, verbose=0)
            return predictions.flatten()
        else:
            # Univariate LSTM
            sequence_length = self.model.input_shape[1] if hasattr(self.model, 'input_shape') else 30
            
            if len(self.builder.historical_df) < sequence_length:
                raise ValueError(f"Not enough historical data. Need at least {sequence_length} points.")
            
            last_sequence = self.builder.historical_df['unit_sales'].tail(sequence_length).values
            
            predictions = []
            current_sequence = last_sequence.copy()
            
            for _ in range(horizon):
                X = current_sequence.reshape(1, sequence_length, 1)
                pred = self.model.predict(X, verbose=0)[0, 0]
                predictions.append(pred)
                current_sequence = np.roll(current_sequence, -1)
                current_sequence[-1] = pred
            
            return np.array(predictions)
    
    def _predict_ml(self, future_df: pd.DataFrame) -> np.ndarray:
        """Predict using ML models (sklearn, xgboost, etc.)."""
        # Try custom forecast method
        if hasattr(self.model, 'forecast_future'):
            try:
                forecast_result = self.model.forecast_future(
                    forecast_horizon=len(future_df),
                    frequency="D",
                    exogenous_future=future_df
                )
                if isinstance(forecast_result, dict) and "point_forecasts" in forecast_result:
                    return np.array(forecast_result["point_forecasts"])
                return np.array(forecast_result)
            except Exception as e:
                st.info(f"Custom forecast method not available: {e}")
        
        # Get available features
        if hasattr(self.model, 'feature_names_in_'):
            available_features = list(self.model.feature_names_in_)
        else:
            available_features = [col for col in future_df.columns 
                                if col != "date" and pd.api.types.is_numeric_dtype(future_df[col])]
        
        # Prepare features
        prediction_features = []
        for feature in available_features:
            if feature in future_df.columns:
                prediction_features.append(feature)
            else:
                if feature in self.builder.historical_df.columns:
                    default_value = self.builder.historical_df[feature].iloc[-1]
                else:
                    default_value = 0
                future_df[feature] = default_value
                prediction_features.append(feature)
        
        # Make predictions
        predictions = self.model.predict(future_df[prediction_features])
        return np.array(predictions)
    
    def get_confidence_intervals(
        self,
        future_df: pd.DataFrame,
        confidence_level: float = 0.95
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Get confidence intervals if available.
        
        Args:
            future_df: DataFrame with future features
            confidence_level: Confidence level (default 0.95)
            
        Returns:
            Tuple of (lower_bound, upper_bound) or None
        """
        try:
            horizon = len(future_df)
            
            # ARIMA/SARIMA models
            if self.model_type in ["arima", "sarima"]:
                if hasattr(self.model, 'get_forecast'):
                    forecast_result = self.model.get_forecast(steps=horizon)
                    conf_int = forecast_result.conf_int(alpha=1-confidence_level)
                    return conf_int.iloc[:, 0].values, conf_int.iloc[:, 1].values
            
            # Prophet models
            elif self.model_type == "prophet":
                prophet_df = future_df.copy()
                if "date" in prophet_df.columns:
                    prophet_df = prophet_df.rename(columns={"date": "ds"})
                
                forecast = self.model.predict(prophet_df)
                return forecast["yhat_lower"].values, forecast["yhat_upper"].values
            
            # ML models with custom uncertainty method
            elif hasattr(self.model, '_simple_uncertainty'):
                from ..utils.helpers import FEATURES
                lower, upper = self.model._simple_uncertainty(
                    pipeline=self.model,
                    X_future=future_df[FEATURES],
                    confidence_level=confidence_level
                )
                return lower, upper
            
            return None
        
        except Exception as e:
            st.info(f"Confidence intervals not available: {e}")
            return None