# app/compoments/model_selector.py
"""
Auto-Best Model Selection Module
Automatically selects the best model based on historical performance metrics
"""

from logging import debug
from app.bootstrap import *  # ensures project root is on sys.path  # noqa: F403
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any  # noqa: F401
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import warnings  # noqa: F401
import traceback
from datetime import timedelta  # noqa: F401

# Import from existing modules
from app.utils.helpers import MODEL_REGISTRY, load_registered_model
from app.components.feature_forecast_builder import FeatureForecastBuilder
from app.ui.forecast_engine import ForecastEngine


class AutoBestModelSelector:
    """Automatically selects the best model based on historical performance."""
    
    def __init__(self, historical_data: pd.DataFrame, validation_split: float = 0.8):
        """
        Initialize model selector.
        
        Args:
            historical_data: Complete historical dataset
            validation_split: Proportion of data to use for training (rest for validation)
        """
        self.historical_data = historical_data.copy()
        self.validation_split = validation_split
        self.validation_results = {}
        self.best_models_cache = {}
        
    def evaluate_single_model(self, model_key: str) -> Optional[Dict[str, float]]:
        """
        Evaluate a single model using time-series cross-validation.
        
        Args:
            model_key: Model key from MODEL_REGISTRY
            
        Returns:
            Dictionary with performance metrics or None if evaluation fails
        """
        try:
            # Get model metadata
            model_meta = MODEL_REGISTRY.get(model_key)
            if not model_meta:
                print(f"❌ Model key '{model_key}' not found in registry")
                return None
            
            print(f"🔍 Evaluating model: {model_meta['label']} ({model_key})")
            
            # Split data chronologically (important for time series)
            split_idx = int(len(self.historical_data) * self.validation_split)
            train_df = self.historical_data.iloc[:split_idx].copy()
            test_df = self.historical_data.iloc[split_idx:].copy()
            
            if len(test_df) < 7:  # Need at least 7 days for meaningful evaluation
                print(f"⚠️ Insufficient test data for {model_key}")
                return None
            
            # Load model
            model = load_registered_model(model_key)
            if model is None:
                print(f"❌ Failed to load model: {model_key}")
                return None
            
            # Create builder for the model type
            model_type = model_meta["model_type"]
            builder = FeatureForecastBuilder(train_df, model_type=model_type)
            
            # Prepare test period features
            test_start = test_df["date"].min()
            horizon = len(test_df)
            
            # Build future features
            future_df = builder.build_future_features(
                start_date=test_start,
                horizon=horizon,
                frequency="D",
                onpromotion=0,  # Assume no promotions during validation
                model_type=model_type
            )
            
            # Initialize forecast engine
            engine = ForecastEngine(model, model_type, builder)
            
            # Generate predictions
            predictions = engine.predict(future_df, horizon, "D")
            
            # Align predictions with test data
            if len(predictions) > len(test_df):
                predictions = predictions[:len(test_df)]
            elif len(predictions) < len(test_df):
                # Pad predictions if needed
                predictions = np.pad(predictions, (0, len(test_df) - len(predictions)), 
                                   mode='edge')
            
            # Calculate metrics
            actual = test_df["unit_sales"].values
            valid_mask = ~np.isnan(actual) & ~np.isnan(predictions) & (actual > 0)
            
            if np.sum(valid_mask) < 5:  # Need at least 5 valid points
                print(f"⚠️ Insufficient valid data points for {model_key}")
                return None
            
            actual_valid = actual[valid_mask]
            predictions_valid = predictions[valid_mask]
            
            metrics = {
                "mae": mean_absolute_error(actual_valid, predictions_valid),
                "rmse": np.sqrt(mean_squared_error(actual_valid, predictions_valid)),
                "mape": mean_absolute_percentage_error(actual_valid, predictions_valid) * 100,
                "bias": np.mean(predictions_valid - actual_valid),
                "correlation": np.corrcoef(actual_valid, predictions_valid)[0, 1] if len(actual_valid) > 1 else 0,
                "r2": 1 - (np.sum((actual_valid - predictions_valid) ** 2) / 
                          np.sum((actual_valid - np.mean(actual_valid)) ** 2)) if len(actual_valid) > 1 else 0,
                "coverage": np.mean((predictions_valid >= actual_valid * 0.9) & 
                                   (predictions_valid <= actual_valid * 1.1)) * 100
            }
            
            # Store results
            self.validation_results[model_key] = {
                **metrics,
                "model_type": model_type,
                "label": model_meta["label"],
                "week": model_meta["week"],
                "test_size": len(test_df),
                "train_size": len(train_df),
                "evaluation_date": pd.Timestamp.now().isoformat()
            }
            
            print(f"✅ Evaluated {model_key}: MAE={metrics['mae']:.2f}, RMSE={metrics['rmse']:.2f}")
            return metrics
            
        except Exception as e:
            print(f"❌ Error evaluating {model_key}: {str(e)}")
            if "debug" in locals() and debug:
                traceback.print_exc()
            return None
    
    def evaluate_all_models(self, max_workers: int = 4, force_reload: bool = False) -> None:
        """
        Evaluate all registered models.
        
        Args:
            max_workers: Maximum number of parallel workers
            force_reload: Force re-evaluation even if cached
        """
        if not force_reload and self.validation_results:
            print("📊 Using cached evaluation results")
            return
        
        print(f"🔬 Evaluating {len(MODEL_REGISTRY)} models...")
        
        # Sequential evaluation (more reliable for complex models)
        for i, model_key in enumerate(MODEL_REGISTRY.keys(), 1):
            print(f"\n[{i}/{len(MODEL_REGISTRY)}] ", end="")
            self.evaluate_single_model(model_key)
        
        print(f"\n✅ Completed evaluation of {len(self.validation_results)} models")
    
    def select_best_models(self, 
                          metric: str = "mae", 
                          top_k: int = 5,
                          model_types: Optional[List[str]] = None,
                          weeks: Optional[List[int]] = None) -> List[Dict[str, Any]]:
        """
        Select best models based on specified criteria.
        
        Args:
            metric: Metric to optimize (mae, rmse, mape, r2, correlation)
            top_k: Number of top models to return
            model_types: Filter by model types (e.g., ['arima', 'random_forest'])
            weeks: Filter by week (e.g., [2, 3])
            
        Returns:
            List of best model configurations
        """
        # Ensure models are evaluated
        if not self.validation_results:
            print("📊 No evaluation results found. Running evaluation...")
            self.evaluate_all_models()
        
        # Filter valid results
        valid_results = {}
        for model_key, results in self.validation_results.items():
            # Check if metric exists and is valid
            if metric not in results or np.isnan(results[metric]):
                continue
            
            # Filter by model type if specified
            if model_types and results["model_type"] not in model_types:
                continue
            
            # Filter by week if specified
            if weeks and results.get("week") not in weeks:
                continue
            
            valid_results[model_key] = results
        
        if not valid_results:
            print("⚠️ No valid results found with the specified filters")
            return []
        
        # Determine sort direction
        lower_is_better = metric in ["mae", "rmse", "mape", "bias"]
        reverse = not lower_is_better  # Reverse for metrics where higher is better
        
        # Sort models by selected metric
        sorted_models = sorted(valid_results.items(),
                              key=lambda x: x[1][metric],
                              reverse=reverse)[:top_k]
        
        # Prepare result
        best_models = []
        for rank, (model_key, results) in enumerate(sorted_models, 1):
            best_models.append({
                "rank": rank,
                "model_key": model_key,
                "model_type": results["model_type"],
                "label": results["label"],
                "week": results["week"],
                "score": results[metric],
                "metrics": {k: v for k, v in results.items() 
                          if k in ['mae', 'rmse', 'mape', 'r2', 'correlation', 'coverage']},
                "test_size": results.get("test_size", 0),
                "train_size": results.get("train_size", 0)
            })
        
        # Cache results
        cache_key = f"{metric}_{top_k}_{model_types}_{weeks}"
        self.best_models_cache[cache_key] = best_models
        
        return best_models
    
    def get_model_recommendation(self, 
                                business_context: str = "general",
                                horizon: int = 30) -> Dict[str, Any]:
        """
        Get model recommendation based on business context.
        
        Args:
            business_context: Type of forecast needed
                             Options: 'general', 'short_term', 'long_term', 
                                     'promotional', 'inventory', 'financial'
            horizon: Forecast horizon in days
            
        Returns:
            Recommended model configuration
        """
        # Define context-specific preferences
        context_preferences = {
            "general": {"metric": "mae", "model_types": None, "weeks": None},
            "short_term": {"metric": "mae", "model_types": ["arima", "sarima", "lstm"], "weeks": None},
            "long_term": {"metric": "mape", "model_types": ["prophet", "ets", "random_forest", "xgboost"], "weeks": None},
            "promotional": {"metric": "coverage", "model_types": ["random_forest", "xgboost", "linear"], "weeks": [3]},
            "inventory": {"metric": "mae", "model_types": None, "weeks": None},
            "financial": {"metric": "mape", "model_types": ["prophet", "ets"], "weeks": [2]}
        }
        
        # Get context preferences
        prefs = context_preferences.get(business_context, context_preferences["general"])
        
        # Adjust based on horizon
        if horizon <= 7:
            prefs["model_types"] = ["arima", "sarima", "lstm"]
        elif horizon >= 90:
            prefs["model_types"] = ["prophet", "ets", "random_forest", "xgboost"]
        
        # Get best models for this context
        best_models = self.select_best_models(
            metric=prefs["metric"],
            top_k=3,
            model_types=prefs["model_types"],
            weeks=prefs["weeks"]
        )
        
        if not best_models:
            # Fallback to general recommendation
            best_models = self.select_best_models(metric="mae", top_k=1)
        
        if best_models:
            recommendation = best_models[0]
            recommendation["context"] = business_context
            recommendation["reason"] = self._generate_recommendation_reason(recommendation, business_context, horizon)
            return recommendation
        
        return {}
    
    def _generate_recommendation_reason(self, model_info: Dict, context: str, horizon: int) -> str:
        """Generate human-readable recommendation reason."""
        reasons = []
        
        # Performance-based reason
        if model_info["metrics"].get("mae", float('inf')) < 10:
            reasons.append("excellent accuracy (MAE < 10)")
        elif model_info["metrics"].get("mae", float('inf')) < 20:
            reasons.append("good accuracy")
        
        # Context-specific reasons
        if context == "short_term":
            reasons.append("optimized for short-term predictions")
        elif context == "long_term":
            reasons.append("stable for long-term forecasting")
        elif context == "promotional":
            reasons.append("handles promotional periods well")
        
        # Model-type reasons
        model_type = model_info["model_type"]
        if model_type in ["arima", "sarima"]:
            reasons.append("captures temporal dependencies effectively")
        elif model_type == "lstm":
            reasons.append("learns complex temporal patterns")
        elif model_type in ["random_forest", "xgboost"]:
            reasons.append("robust to outliers and captures non-linear relationships")
        elif model_type == "prophet":
            reasons.append("excellent for seasonal patterns and holidays")
        
        # Horizon reason
        if horizon <= 7:
            reasons.append("optimized for very short horizons")
        elif horizon >= 90:
            reasons.append("performs well on long horizons")
        
        return f"Selected for: {', '.join(reasons)}"
    
    def create_comparison_report(self) -> pd.DataFrame:
        """Create comprehensive comparison report of all models."""
        if not self.validation_results:
            self.evaluate_all_models()
        
        # Convert to DataFrame
        rows = []
        for model_key, results in self.validation_results.items():
            row = {
                "model_key": model_key,
                "label": results["label"],
                "model_type": results["model_type"],
                "week": results.get("week", "N/A"),
            }
            
            # Add all metrics
            for metric in ['mae', 'rmse', 'mape', 'r2', 'correlation', 'coverage', 'bias']:
                if metric in results:
                    row[metric] = results[metric]
            
            rows.append(row)
        
        df = pd.DataFrame(rows)
        
        # Sort by MAE (primary), then RMSE (secondary)
        if 'mae' in df.columns and 'rmse' in df.columns:
            df = df.sort_values(['mae', 'rmse'])
        
        return df
    
    def visualize_comparison(self):
        """Create visualization of model comparison."""
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        
        df = self.create_comparison_report()
        
        if df.empty:
            print("No data to visualize")
            return None
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("MAE Comparison", "RMSE Comparison", 
                           "MAPE Comparison", "R² Comparison"),
            vertical_spacing=0.15,
            horizontal_spacing=0.15
        )
        
        metrics = ['mae', 'rmse', 'mape', 'r2']
        positions = [(1, 1), (1, 2), (2, 1), (2, 2)]
        
        for metric, (row, col) in zip(metrics, positions):
            if metric in df.columns:
                # Sort by metric value
                metric_df = df.sort_values(metric, ascending=metric != 'r2')
                
                fig.add_trace(
                    go.Bar(
                        x=metric_df['label'],
                        y=metric_df[metric],
                        text=[f"{v:.3f}" for v in metric_df[metric]],
                        textposition='outside',
                        marker_color='lightblue' if metric != 'r2' else 'lightgreen',
                        name=metric.upper()
                    ),
                    row=row, col=col
                )
                
                fig.update_xaxes(title_text="Model", row=row, col=col, tickangle=45)
                fig.update_yaxes(title_text=metric.upper(), row=row, col=col)
        
        fig.update_layout(
            title="Model Performance Comparison",
            height=800,
            showlegend=False,
            plot_bgcolor='white'
        )
        
        return fig
    
    def save_evaluation_results(self, filepath: str = "model_evaluation_results.csv"):
        """Save evaluation results to CSV file."""
        df = self.create_comparison_report()
        
        if not df.empty:
            df.to_csv(filepath, index=False)
            print(f"✅ Results saved to {filepath}")
            return True
        
        return False