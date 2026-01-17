# app/components/feature_forecast_builder.py
"""
Feature Builder for Future Forecasting
Handles feature engineering for different model types and frequencies
"""

from app.bootstrap import *  # ensures project root is on sys.path  # noqa: F403
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import warnings



class FeatureForecastBuilder:
    """
    Builder for future forecast features with support for multiple frequencies
    and model types.
    """

    def __init__(self, historical_df: pd.DataFrame, model_type: str = "arima"):
        """
        Initialize with historical data.

        Args:
            historical_df: DataFrame with historical time series data
        """
        self.historical_df = historical_df.copy()
        self.original_features = list(historical_df.columns)
        self.model_type = model_type  

    def build_future_features(
        self,
        start_date: pd.Timestamp,
        horizon: int,
        frequency: str = "D",
        onpromotion: int = 0,
        model_type: str = "ml",
        include_lags: bool = True,
        include_rolling: bool = True,
        include_seasonal: bool = True,
        include_cyclical: bool = True
    ) -> pd.DataFrame:
        """
        Build future feature matrix for forecasting.

        Args:
            start_date: Start date for forecast
            horizon: Number of periods to forecast
            frequency: Frequency ('D', 'W', 'M', 'Q', 'Y')
            onpromotion: Promotion status (0 or 1)
            model_type: Type of model ('ml', 'arima', 'prophet', 'lstm')
            include_lags: Whether to include lag features
            include_rolling: Whether to include rolling features
            include_seasonal: Whether to include seasonal features
            include_cyclical: Whether to include cyclical features

        Returns:
            DataFrame with future features
        """
        # Update model_type attribute
        self.model_type = model_type

        # Rest of the method remains unchanged
        dates = self._generate_future_dates(start_date, horizon, frequency)
        future_df = pd.DataFrame({"date": dates})

        if model_type in ["arima", "sarima", "ets"]:
            future_df = self._add_basic_features(future_df, onpromotion)
            return future_df

        future_df = self._add_basic_features(future_df, onpromotion)

        if include_seasonal:
            future_df = self._add_time_features(future_df, frequency)

        if include_cyclical:
            future_df = self._add_cyclical_features(future_df, frequency)

        if include_lags:
            future_df = self._add_lag_features(future_df, horizon)

        if include_rolling:
            future_df = self._add_rolling_features(future_df)

        future_df = self._add_static_features(future_df)
        future_df = self._adapt_for_model_type(future_df, model_type)
        future_df = self._ensure_complete_features(future_df)

        return future_df

    
    def _generate_future_dates(
        self, 
        start_date: pd.Timestamp, 
        horizon: int, 
        frequency: str
    ) -> pd.DatetimeIndex:
        """
        Generate future dates based on frequency.
        """
        freq_map = {
            "D": "D",
            "W": "W",
            "M": "MS",  # Month start
            "Q": "QS",  # Quarter start
            "Y": "YS"   # Year start
        }
        
        if frequency not in freq_map:
            warnings.warn(f"Unknown frequency '{frequency}', defaulting to daily")
            frequency = "D"
        
        return pd.date_range(
            start=start_date, 
            periods=horizon, 
            freq=freq_map[frequency]
        )
    
    def _add_basic_features(
        self, 
        future_df: pd.DataFrame, 
        onpromotion: int
    ) -> pd.DataFrame:
        """
        Add basic features to future DataFrame.
        """
        df = future_df.copy()
        df["onpromotion"] = onpromotion
        
        # Add promotion lag if it exists in historical data
        if "onpromotion_lag_1" in self.historical_df.columns:
            df["onpromotion_lag_1"] = self.historical_df["onpromotion"].iloc[-1]
        
        return df
    
    def _add_time_features(
        self, 
        future_df: pd.DataFrame, 
        frequency: str
    ) -> pd.DataFrame:
        """
        Add time-based seasonal features.
        """
        df = future_df.copy()
        
        # Basic time features
        df["day_of_week"] = df["date"].dt.dayofweek
        df["day_of_month"] = df["date"].dt.day
        df["day_of_year"] = df["date"].dt.dayofyear
        df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
        df["month"] = df["date"].dt.month
        df["quarter"] = df["date"].dt.quarter
        df["year"] = df["date"].dt.year
        
        # Week/month/year fractions
        df["week_fraction"] = df["week_of_year"] / 52
        df["month_fraction"] = df["month"] / 12
        df["quarter_fraction"] = df["quarter"] / 4
        
        # Business days
        df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
        df["is_month_start"] = df["date"].dt.is_month_start.astype(int)
        df["is_month_end"] = df["date"].dt.is_month_end.astype(int)
        df["is_quarter_start"] = df["date"].dt.is_quarter_start.astype(int)
        df["is_quarter_end"] = df["date"].dt.is_quarter_end.astype(int)
        df["is_year_start"] = df["date"].dt.is_year_start.astype(int)
        df["is_year_end"] = df["date"].dt.is_year_end.astype(int)
        
        # Seasonality based on frequency
        if frequency == "W":
            df["week_sin"] = np.sin(2 * np.pi * df["week_fraction"])
            df["week_cos"] = np.cos(2 * np.pi * df["week_fraction"])
        
        return df
    
    def _add_cyclical_features(
        self, 
        future_df: pd.DataFrame, 
        frequency: str
    ) -> pd.DataFrame:
        """
        Add cyclical features for seasonality.
        """
        df = future_df.copy()
        
        # Daily seasonality
        if "day_of_week" in df.columns:
            df["day_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
            df["day_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
            df["day_of_week_sin"] = df["day_sin"]  # Alias for compatibility
            df["day_of_week_cos"] = df["day_cos"]  # Alias for compatibility
        
        # Monthly seasonality
        if "month" in df.columns:
            df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
            df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
        
        # Yearly seasonality
        if "day_of_year" in df.columns:
            df["year_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 365)
            df["year_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 365)
        
        # Quarterly seasonality
        if "quarter" in df.columns:
            df["quarter_sin"] = np.sin(2 * np.pi * df["quarter"] / 4)
            df["quarter_cos"] = np.cos(2 * np.pi * df["quarter"] / 4)
        
        return df
    
    def _add_lag_features(
        self, 
        future_df: pd.DataFrame, 
        horizon: int
    ) -> pd.DataFrame:
        """
        Add lag features to future DataFrame.
        """
        df = future_df.copy()
        
        # Common lag periods
        lag_periods = [1, 3, 7, 14, 30, 90, 180, 365]
        
        for lag in lag_periods:
            col_name = f"unit_sales_lag_{lag}"
            if col_name in self.historical_df.columns:
                # Use last available value from historical data
                if len(self.historical_df) >= lag:
                    last_value = self.historical_df["unit_sales"].iloc[-lag]
                else:
                    last_value = self.historical_df["unit_sales"].mean()
                
                df[col_name] = last_value
        
        # For multivariate LSTM or other models that need lag sequences
        if "unit_sales" in self.historical_df.columns:
            # Add recent sequence as features
            recent_sequence = self.historical_df["unit_sales"].tail(30).values
            for i in range(min(30, len(recent_sequence))):
                df[f"recent_{i+1}"] = recent_sequence[i] if i < len(recent_sequence) else 0
        
        return df
    
    def _add_rolling_features(self, future_df: pd.DataFrame) -> pd.DataFrame:
        """
        Add rolling window features.
        """
        df = future_df.copy()
        
        if "unit_sales" not in self.historical_df.columns:
            return df
        
        # Rolling mean periods
        mean_periods = [3, 7, 14, 30, 90, 365]
        
        for period in mean_periods:
            mean_col = f"unit_sales_r{period}_mean"
            median_col = f"unit_sales_r{period}_median"
            std_col = f"unit_sales_r{period}_std"
            
            # Calculate from historical data
            if len(self.historical_df) >= period:
                mean_val = self.historical_df["unit_sales"].tail(period).mean()
                median_val = self.historical_df["unit_sales"].tail(period).median()
                std_val = self.historical_df["unit_sales"].tail(period).std()
            else:
                mean_val = self.historical_df["unit_sales"].mean()
                median_val = self.historical_df["unit_sales"].median()
                std_val = self.historical_df["unit_sales"].std()
            
            df[mean_col] = mean_val
            df[median_col] = median_val
            df[std_col] = std_val
        
        # Expanding features
        if len(self.historical_df) > 0:
            df["unit_sales_expanding_mean"] = self.historical_df["unit_sales"].mean()
            df["unit_sales_expanding_std"] = self.historical_df["unit_sales"].std()
        
        return df
    
    def _add_static_features(self, future_df: pd.DataFrame) -> pd.DataFrame:
        """
        Add static features from historical data.
        """
        df = future_df.copy()
        
        # Static features to carry forward
        static_features = [
            "store_avg_sales", "store_item_median", "store_item_mean",
            "days_until_holiday", "days_since_holiday", "holiday_flag",
            "is_holiday", "is_holiday_eve", "is_holiday_week"
        ]
        
        for feature in static_features:
            if feature in self.historical_df.columns:
                # Use last available value
                last_value = self.historical_df[feature].iloc[-1]
                df[feature] = last_value
        
        return df
    
    def _adapt_for_model_type(
        self, 
        future_df: pd.DataFrame, 
        model_type: str
    ) -> pd.DataFrame:
        """
        Adapt features for specific model types.
        """
        df = future_df.copy()
        
        # Prophet requires specific column names
        if model_type == "prophet":
            if "date" in df.columns:
                df = df.rename(columns={"date": "ds"})
        
        # ARIMA/SARIMA may need differenced features
        elif model_type in ["arima", "sarima"]:
            # For ARIMA models, we mainly need the date and basic features
            # Keep only essential columns
            essential_cols = ["date", "onpromotion"]
            if "onpromotion_lag_1" in df.columns:
                essential_cols.append("onpromotion_lag_1")
            
            # Keep only essential columns and any features that might be needed
            df = df[essential_cols]
        
        # LSTM may require sequence reshaping
        elif model_type == "lstm":
            # Ensure sequence features are present
            sequence_length = 30  # Default, can be parameterized
            for i in range(sequence_length):
                seq_col = f"sequence_{i}"
                if seq_col not in df.columns:
                    # Initialize with zeros or last value
                    df[seq_col] = 0
        
        return df
    
    def _ensure_complete_features(self, future_df: pd.DataFrame) -> pd.DataFrame:
        """
        Ensure all columns from historical data are present in future DataFrame.
        """
        df = future_df.copy()
        
        # For ARIMA models, don't add all features
        if self.model_type in ["arima", "sarima", "ets"]:
            return df
        
        # Add missing columns with default values
        for col in self.original_features:
            if col not in df.columns and col != "date":
                # Try to use the last value from historical data
                if col in self.historical_df.columns:
                    default_value = self.historical_df[col].iloc[-1]
                else:
                    default_value = 0
                
                df[col] = default_value
        
        # Reorder columns to match historical data
        column_order = [col for col in self.original_features if col in df.columns]
        remaining_cols = [col for col in df.columns if col not in column_order]
        df = df[column_order + remaining_cols]
        
        return df
    
    def build_sequence_for_lstm(
        self,
        sequence_length: int = 30,
        target_col: str = "unit_sales",
        feature_cols: Optional[List[str]] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build sequences for LSTM models.
        
        Args:
            sequence_length: Length of input sequences
            target_col: Name of target column
            feature_cols: List of feature columns (None = all except target)
            
        Returns:
            Tuple of (X_sequences, y_targets)
        """
        df = self.historical_df.copy()
        
        if feature_cols is None:
            feature_cols = [col for col in df.columns if col != target_col]
        
        # Prepare sequences
        X_sequences = []
        y_targets = []
        
        for i in range(len(df) - sequence_length):
            # Extract sequence
            sequence = df.iloc[i:i + sequence_length][feature_cols].values
            target = df.iloc[i + sequence_length][target_col]
            
            X_sequences.append(sequence)
            y_targets.append(target)
        
        return np.array(X_sequences), np.array(y_targets)
    
    def build_future_sequence_for_lstm(
        self,
        future_df: pd.DataFrame,
        sequence_length: int = 30,
        feature_cols: Optional[List[str]] = None
    ) -> np.ndarray:
        """
        Build future sequences for LSTM prediction.
        """
        if feature_cols is None:
            feature_cols = [col for col in future_df.columns if col != "date"]
        
        # Combine last sequence_length-1 rows from historical with future
        combined = pd.concat([
            self.historical_df.tail(sequence_length - 1),
            future_df
        ])
        
        # Build sequences
        sequences = []
        for i in range(len(future_df)):
            start_idx = i
            end_idx = i + sequence_length
            sequence = combined.iloc[start_idx:end_idx][feature_cols].values
            sequences.append(sequence)
        
        return np.array(sequences)
    
    def get_feature_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about features in the historical data.
        
        Returns:
            Dictionary with feature categorization and statistics
        """
        features = list(self.historical_df.columns)
        
        # Categorize features
        numeric_features = []
        lag_features = []
        rolling_features = []
        seasonal_features = []
        date_features = []
        categorical_features = []
        other_features = []
        
        for col in features:
            col_lower = col.lower()
            
            if col_lower == "date" or "date" in col_lower:
                date_features.append(col)
            elif "lag" in col_lower:
                lag_features.append(col)
            elif any(x in col_lower for x in ["mean", "median", "std", "r3", "r7", "r14", "r30", "r90", "r365", "expanding"]):
                rolling_features.append(col)
            elif any(x in col_lower for x in ["month", "year", "day", "week", "quarter", "season", "sin", "cos"]):
                seasonal_features.append(col)
            elif self.historical_df[col].dtype in ['int64', 'float64']:
                numeric_features.append(col)
            elif self.historical_df[col].dtype == 'object' or self.historical_df[col].dtype.name == 'category':
                categorical_features.append(col)
            else:
                other_features.append(col)
        
        # Calculate statistics
        stats = {
            "total_features": len(features),
            "numeric_features": numeric_features,
            "lag_features": lag_features,
            "rolling_features": rolling_features,
            "seasonal_features": seasonal_features,
            "date_features": date_features,
            "categorical_features": categorical_features,
            "other_features": other_features,
            "feature_counts": {
                "numeric": len(numeric_features),
                "lag": len(lag_features),
                "rolling": len(rolling_features),
                "seasonal": len(seasonal_features),
                "date": len(date_features),
                "categorical": len(categorical_features),
                "other": len(other_features)
            }
        }
        
        # Add data shape info
        stats["data_shape"] = {
            "rows": len(self.historical_df),
            "columns": len(features),
            "memory_mb": self.historical_df.memory_usage(deep=True).sum() / (1024 * 1024)
        }
        
        # Add date range if available
        if date_features:
            date_col = date_features[0]
            try:
                stats["date_range"] = {
                    "start": str(self.historical_df[date_col].min()),
                    "end": str(self.historical_df[date_col].max()),
                    "days": (self.historical_df[date_col].max() - self.historical_df[date_col].min()).days
                }
            except:  # noqa: E722
                pass
        
        return stats