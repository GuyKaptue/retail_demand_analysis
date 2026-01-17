# app/components/performance_tracker.py

"""
Model Performance Tracking Module
Tracks forecast performance over time for continuous improvement
"""

from app.bootstrap import *  # ensures project root is on sys.path  # noqa: F403
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any  # noqa: F401
import warnings  # noqa: F401
from dataclasses import dataclass, asdict
from enum import Enum


class ForecastStatus(Enum):
    """Status of forecast evaluation."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    VALIDATED = "validated"
    OBSOLETE = "obsolete"


@dataclass
class ForecastRecord:
    """Data class for forecast run records."""
    id: str
    timestamp: str
    model_key: str
    model_type: str
    model_label: str
    horizon: int
    frequency: str
    start_date: str
    end_date: str
    total_forecast: float
    average_forecast: float
    confidence_level: float
    status: str
    metrics: Dict[str, float]
    config: Dict[str, Any]
    file_path: str
    notes: str = ""


class ModelPerformanceTracker:
    """Tracks model performance over time for continuous improvement."""
    
    def __init__(self, 
                 log_dir: str = "data/model_performance",
                 max_records_per_model: int = 100):
        """
        Initialize performance tracker.
        
        Args:
            log_dir: Directory to store performance logs
            max_records_per_model: Maximum records to keep per model
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.max_records = max_records_per_model
        
        # Initialize tracking files
        self.records_file = self.log_dir / "forecast_records.json"
        self.rankings_file = self.log_dir / "model_rankings.json"
        self.metrics_file = self.log_dir / "performance_metrics.json"
        
        # Load existing data
        self.records = self._load_records()
        self.rankings = self._load_rankings()
        self.metrics_history = self._load_metrics_history()
    
    def _load_records(self) -> Dict[str, List[ForecastRecord]]:
        """Load forecast records from file."""
        if self.records_file.exists():
            try:
                with open(self.records_file, 'r') as f:
                    data = json.load(f)
                
                # Convert back to ForecastRecord objects
                records = {}
                for model_key, model_records in data.items():
                    records[model_key] = [
                        ForecastRecord(**record) for record in model_records
                    ]
                return records
            except Exception as e:
                print(f"Warning: Could not load records: {e}")
        
        return {}
    
    def _load_rankings(self) -> Dict[str, Any]:
        """Load model rankings from file."""
        if self.rankings_file.exists():
            try:
                with open(self.rankings_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Warning: Could not load rankings: {e}")
        
        return {}
    
    def _load_metrics_history(self) -> pd.DataFrame:
        """Load metrics history from file."""
        if self.metrics_file.exists():
            try:
                df = pd.read_json(self.metrics_file)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                return df
            except Exception as e:
                print(f"Warning: Could not load metrics history: {e}")
        
        return pd.DataFrame()
    
    def _save_records(self):
        """Save forecast records to file."""
        try:
            # Convert records to serializable format
            serializable_records = {}
            for model_key, model_records in self.records.items():
                serializable_records[model_key] = [
                    asdict(record) for record in model_records
                ]
            
            with open(self.records_file, 'w') as f:
                json.dump(serializable_records, f, indent=2, default=str)
        except Exception as e:
            print(f"Error saving records: {e}")
    
    def _save_rankings(self):
        """Save model rankings to file."""
        try:
            with open(self.rankings_file, 'w') as f:
                json.dump(self.rankings, f, indent=2)
        except Exception as e:
            print(f"Error saving rankings: {e}")
    
    def _save_metrics_history(self):
        """Save metrics history to file."""
        try:
            if not self.metrics_history.empty:
                self.metrics_history.to_json(self.metrics_file, indent=2)
        except Exception as e:
            print(f"Error saving metrics history: {e}")
    
    def log_forecast_run(self,
                        model_key: str,
                        config: Dict[str, Any],
                        forecast_df: pd.DataFrame,
                        actual_df: Optional[pd.DataFrame] = None,
                        notes: str = "") -> str:
        """
        Log a forecast run with optional performance evaluation.
        
        Args:
            model_key: Model identifier
            config: Forecast configuration
            forecast_df: Forecast results DataFrame
            actual_df: Actual values for evaluation (optional)
            notes: Additional notes about the forecast
            
        Returns:
            Forecast record ID
        """
        from app.utils.helpers import MODEL_REGISTRY
        
        # Generate unique ID
        forecast_id = f"{model_key}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Get model metadata
        model_meta = MODEL_REGISTRY.get(model_key, {})
        
        # Calculate forecast summary
        forecast_summary = {
            "total_forecast": float(forecast_df["point_forecast"].sum()),
            "average_forecast": float(forecast_df["point_forecast"].mean()),
            "std_forecast": float(forecast_df["point_forecast"].std()),
            "min_forecast": float(forecast_df["point_forecast"].min()),
            "max_forecast": float(forecast_df["point_forecast"].max()),
            "forecast_horizon": len(forecast_df),
            "start_date": forecast_df["date"].min().isoformat() if "date" in forecast_df.columns else None,
            "end_date": forecast_df["date"].max().isoformat() if "date" in forecast_df.columns else None
        }
        
        # Initialize performance metrics
        performance_metrics = {}
        status = ForecastStatus.COMPLETED
        
        # Calculate performance if actual data is available
        if actual_df is not None and not actual_df.empty:
            try:
                performance_metrics = self._calculate_performance_metrics(
                    forecast_df, actual_df, config.get("confidence_level", 0.95)
                )
                status = ForecastStatus.VALIDATED
            except Exception as e:
                print(f"Warning: Could not calculate performance metrics: {e}")
                performance_metrics = {"error": str(e)}
                status = ForecastStatus.COMPLETED
        
        # Create forecast record
        record = ForecastRecord(
            id=forecast_id,
            timestamp=datetime.now().isoformat(),
            model_key=model_key,
            model_type=model_meta.get("model_type", "unknown"),
            model_label=model_meta.get("label", "Unknown Model"),
            horizon=config.get("horizon", len(forecast_df)),
            frequency=config.get("freq", "D"),
            start_date=config.get("start_date", ""),
            end_date=(pd.to_datetime(config.get("start_date")) + 
                     pd.Timedelta(days=config.get("horizon", 0))).isoformat() 
                     if config.get("start_date") else "",
            total_forecast=forecast_summary["total_forecast"],
            average_forecast=forecast_summary["average_forecast"],
            confidence_level=config.get("confidence_level", 0.95),
            status=status.value,
            metrics=performance_metrics,
            config=config,
            file_path=f"forecasts/{forecast_id}.csv",
            notes=notes
        )
        
        # Save forecast data to CSV
        self._save_forecast_data(forecast_id, forecast_df)
        
        # Update records
        if model_key not in self.records:
            self.records[model_key] = []
        
        self.records[model_key].append(record)
        
        # Keep only recent records
        if len(self.records[model_key]) > self.max_records:
            self.records[model_key] = self.records[model_key][-self.max_records:]
        
        # Update rankings if performance metrics are available
        if performance_metrics and "mae" in performance_metrics:
            self._update_model_rankings(model_key, performance_metrics, record)
        
        # Update metrics history
        self._update_metrics_history(model_key, performance_metrics, record)
        
        # Save all data
        self._save_records()
        self._save_rankings()
        self._save_metrics_history()
        
        print(f"✅ Logged forecast run: {forecast_id}")
        return forecast_id
    
    def _calculate_performance_metrics(self,
                                     forecast_df: pd.DataFrame,
                                     actual_df: pd.DataFrame,
                                     confidence_level: float = 0.95) -> Dict[str, float]:
        """Calculate comprehensive performance metrics."""
        
        # Merge forecast with actual on date
        date_col = "date"
        forecast_col = "point_forecast"
        actual_col = "unit_sales"
        
        # Ensure both DataFrames have date columns
        if date_col not in forecast_df.columns:
            forecast_df = forecast_df.reset_index()
        
        if date_col not in actual_df.columns:
            actual_df = actual_df.reset_index()
        
        # Convert dates to datetime for merging
        forecast_df[date_col] = pd.to_datetime(forecast_df[date_col])
        actual_df[date_col] = pd.to_datetime(actual_df[date_col])
        
        # Merge data
        merged = pd.merge(
            forecast_df[[date_col, forecast_col]],
            actual_df[[date_col, actual_col]],
            on=date_col,
            how="inner",
            suffixes=("_forecast", "_actual")
        )
        
        if merged.empty:
            return {"error": "No overlapping dates for comparison"}
        
        # Calculate errors
        errors = merged[actual_col] - merged[forecast_col]
        absolute_errors = errors.abs()
        percentage_errors = (absolute_errors / merged[actual_col]) * 100
        
        # Basic metrics
        metrics = {
            "mae": float(absolute_errors.mean()),
            "rmse": float(np.sqrt((errors ** 2).mean())),
            "mape": float(percentage_errors.mean()),
            "bias": float(errors.mean()),
            "std_error": float(errors.std()),
            "correlation": float(np.corrcoef(merged[actual_col], merged[forecast_col])[0, 1]) 
                          if len(merged) > 1 else 0.0,
            "r2": float(1 - (np.sum(errors ** 2) / 
                            np.sum((merged[actual_col] - merged[actual_col].mean()) ** 2))) 
                 if len(merged) > 1 else 0.0
        }
        
        # Directional accuracy
        forecast_changes = merged[forecast_col].pct_change().fillna(0)
        actual_changes = merged[actual_col].pct_change().fillna(0)
        
        same_direction = ((forecast_changes > 0) == (actual_changes > 0)).mean()
        metrics["direction_accuracy"] = float(same_direction * 100)
        
        # Coverage metrics if confidence intervals available
        if "lower_bound" in forecast_df.columns and "upper_bound" in forecast_df.columns:
            merged_with_bounds = pd.merge(
                forecast_df[[date_col, forecast_col, "lower_bound", "upper_bound"]],
                actual_df[[date_col, actual_col]],
                on=date_col,
                how="inner"
            )
            
            if not merged_with_bounds.empty:
                coverage = ((merged_with_bounds[actual_col] >= merged_with_bounds["lower_bound"]) & 
                          (merged_with_bounds[actual_col] <= merged_with_bounds["upper_bound"])).mean()
                metrics["coverage"] = float(coverage * 100)
                
                # Average width of confidence interval
                avg_width = (merged_with_bounds["upper_bound"] - merged_with_bounds["lower_bound"]).mean()
                metrics["avg_ci_width"] = float(avg_width)
                metrics["ci_width_pct"] = float(avg_width / merged_with_bounds[actual_col].mean() * 100)
        
        # Time-based metrics
        metrics["n_points"] = len(merged)
        metrics["forecast_horizon_actual"] = len(merged)
        
        return metrics
    
    def _save_forecast_data(self, forecast_id: str, forecast_df: pd.DataFrame):
        """Save forecast data to CSV file."""
        forecast_dir = self.log_dir / "forecasts"
        forecast_dir.mkdir(exist_ok=True)
        
        file_path = forecast_dir / f"{forecast_id}.csv"
        forecast_df.to_csv(file_path, index=False)
    
    def _update_model_rankings(self, 
                              model_key: str, 
                              metrics: Dict[str, float],
                              record: ForecastRecord):
        """Update model performance rankings."""
        
        if model_key not in self.rankings:
            self.rankings[model_key] = {
                "total_runs": 0,
                "successful_runs": 0,
                "failed_runs": 0,
                "total_mae": 0.0,
                "total_rmse": 0.0,
                "total_mape": 0.0,
                "best_mae": float('inf'),
                "best_rmse": float('inf'),
                "best_mape": float('inf'),
                "worst_mae": 0.0,
                "worst_rmse": 0.0,
                "worst_mape": 0.0,
                "last_run": None,
                "last_metrics": {},
                "avg_horizon": 0,
                "reliability_score": 0.0
            }
        
        stats = self.rankings[model_key]
        stats["total_runs"] += 1
        
        if "error" not in metrics:
            stats["successful_runs"] += 1
        else:
            stats["failed_runs"] += 1
        
        # Update metrics
        for metric in ['mae', 'rmse', 'mape']:
            if metric in metrics:
                value = metrics[metric]
                stats[f"total_{metric}"] += value
                stats[f"best_{metric}"] = min(stats[f"best_{metric}"], value)
                stats[f"worst_{metric}"] = max(stats[f"worst_{metric}"], value)
        
        # Update horizon tracking
        stats["avg_horizon"] = (stats["avg_horizon"] * (stats["total_runs"] - 1) + 
                               record.horizon) / stats["total_runs"]
        
        # Update last run info
        stats["last_run"] = record.timestamp
        stats["last_metrics"] = metrics
        
        # Calculate reliability score
        if stats["total_runs"] > 0:
            success_rate = stats["successful_runs"] / stats["total_runs"]
            if "mae" in metrics and stats["best_mae"] != float('inf'):
                mae_score = 1 - (metrics["mae"] / max(stats["best_mae"] * 2, 1))
                stats["reliability_score"] = (success_rate * 0.4 + mae_score * 0.6) * 100
    
    def _update_metrics_history(self, 
                               model_key: str, 
                               metrics: Dict[str, float],
                               record: ForecastRecord):
        """Update historical metrics DataFrame."""
        
        history_entry = {
            "timestamp": pd.Timestamp(record.timestamp),
            "model_key": model_key,
            "model_type": record.model_type,
            "model_label": record.model_label,
            "horizon": record.horizon,
            "frequency": record.frequency,
            "confidence_level": record.confidence_level,
            "status": record.status,
            **metrics
        }
        
        # Convert to DataFrame and append
        new_row = pd.DataFrame([history_entry])
        
        if self.metrics_history.empty:
            self.metrics_history = new_row
        else:
            self.metrics_history = pd.concat([self.metrics_history, new_row], ignore_index=True)
        
        # Keep only recent history (last 1000 records)
        if len(self.metrics_history) > 1000:
            self.metrics_history = self.metrics_history.tail(1000)
    
    def get_model_performance(self, model_key: str) -> Dict[str, Any]:
        """Get performance statistics for a specific model."""
        
        if model_key not in self.rankings:
            return {"error": f"No performance data for model: {model_key}"}
        
        stats = self.rankings[model_key].copy()
        
        # Calculate averages
        if stats["total_runs"] > 0:
            for metric in ['mae', 'rmse', 'mape']:
                total = stats.get(f"total_{metric}", 0)
                stats[f"avg_{metric}"] = total / stats["successful_runs"] if stats["successful_runs"] > 0 else 0
        
        # Get recent runs
        recent_runs = []
        if model_key in self.records:
            recent_runs = [asdict(r) for r in self.records[model_key][-5:]]
        
        stats["recent_runs"] = recent_runs
        stats["success_rate"] = (stats["successful_runs"] / stats["total_runs"] * 100 
                                if stats["total_runs"] > 0 else 0)
        
        return stats
    
    def get_top_models(self, 
                      metric: str = "reliability_score",
                      top_n: int = 5,
                      min_runs: int = 3,
                      model_types: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Get top performing models based on specified criteria.
        
        Args:
            metric: Metric to sort by
            top_n: Number of top models to return
            min_runs: Minimum number of runs required
            model_types: Filter by model types
            
        Returns:
            List of top model statistics
        """
        valid_models = {}
        
        for model_key, stats in self.rankings.items():
            # Apply filters
            if stats["total_runs"] < min_runs:
                continue
            
            if model_types and stats.get("model_type", "unknown") not in model_types:
                continue
            
            # Get model metadata
            from app.utils.helpers import MODEL_REGISTRY
            model_meta = MODEL_REGISTRY.get(model_key, {})
            
            valid_models[model_key] = {
                **stats,
                "model_key": model_key,
                "model_type": stats.get("model_type", model_meta.get("model_type", "unknown")),
                "model_label": model_meta.get("label", "Unknown Model"),
                "week": model_meta.get("week", 0)
            }
        
        if not valid_models:
            return []
        
        # Determine sort direction
        lower_is_better = metric in ["mae", "rmse", "mape", "best_mae", "best_rmse", "best_mape"]
        
        # Sort models
        sorted_models = sorted(
            valid_models.items(),
            key=lambda x: x[1].get(metric, float('inf') if lower_is_better else 0),
            reverse=not lower_is_better
        )[:top_n]
        
        # Format results
        top_models = []
        for rank, (model_key, stats) in enumerate(sorted_models, 1):
            top_models.append({
                "rank": rank,
                "model_key": model_key,
                "model_label": stats["model_label"],
                "model_type": stats["model_type"],
                "week": stats["week"],
                "reliability_score": stats.get("reliability_score", 0),
                "success_rate": stats.get("success_rate", 0),
                "avg_mae": stats.get("avg_mae", 0),
                "avg_rmse": stats.get("avg_rmse", 0),
                "total_runs": stats["total_runs"],
                "successful_runs": stats["successful_runs"],
                "best_mae": stats.get("best_mae", 0),
                "last_run": stats.get("last_run", "")
            })
        
        return top_models
    
    def get_performance_trends(self, 
                              model_key: str,
                              metric: str = "mae",
                              window_days: int = 30) -> pd.DataFrame:
        """
        Get performance trends for a model over time.
        
        Args:
            model_key: Model identifier
            metric: Metric to analyze
            window_days: Rolling window in days
            
        Returns:
            DataFrame with performance trends
        """
        if self.metrics_history.empty:
            return pd.DataFrame()
        
        # Filter for specific model
        model_history = self.metrics_history[
            self.metrics_history["model_key"] == model_key
        ].copy()
        
        if model_history.empty:
            return pd.DataFrame()
        
        # Sort by timestamp
        model_history = model_history.sort_values("timestamp")
        
        # Calculate rolling metrics
        if metric in model_history.columns:
            model_history[f"{metric}_rolling"] = (
                model_history[metric].rolling(window=min(7, len(model_history)), 
                                             min_periods=1).mean()
            )
        
        return model_history
    
    def generate_performance_report(self, 
                                   output_file: str = "performance_report.html") -> str:
        """
        Generate comprehensive HTML performance report.
        
        Args:
            output_file: Output file path
            
        Returns:
            Path to generated report
        """
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        
        # Get top models
        top_models = self.get_top_models(top_n=10)
        
        # Create figure
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("Top Models by Reliability", 
                          "Performance Trends",
                          "Success Rates", 
                          "Model Run Distribution"),
            vertical_spacing=0.15,
            horizontal_spacing=0.15
        )
        
        # 1. Top models bar chart
        if top_models:
            model_labels = [m["model_label"] for m in top_models]
            reliability_scores = [m["reliability_score"] for m in top_models]
            
            fig.add_trace(
                go.Bar(
                    x=model_labels,
                    y=reliability_scores,
                    name="Reliability Score",
                    marker_color='lightblue',
                    text=[f"{s:.1f}%" for s in reliability_scores],
                    textposition='outside'
                ),
                row=1, col=1
            )
        
        # 2. Performance trends (if we have history)
        if not self.metrics_history.empty:
            # Get recent metrics
            recent = self.metrics_history.tail(50)
            
            fig.add_trace(
                go.Scatter(
                    x=recent["timestamp"],
                    y=recent["mae"],
                    mode="lines+markers",
                    name="MAE Trend",
                    line=dict(color="red", width=2)
                ),
                row=1, col=2
            )
        
        # 3. Success rates
        success_rates = []
        model_names = []
        
        for model_key in list(self.rankings.keys())[:10]:
            stats = self.rankings[model_key]
            if stats["total_runs"] > 0:
                success_rate = (stats["successful_runs"] / stats["total_runs"]) * 100
                success_rates.append(success_rate)
                
                from app.utils.helpers import MODEL_REGISTRY
                model_meta = MODEL_REGISTRY.get(model_key, {})
                model_names.append(model_meta.get("label", model_key))
        
        if success_rates:
            fig.add_trace(
                go.Bar(
                    x=model_names,
                    y=success_rates,
                    name="Success Rate",
                    marker_color='green',
                    text=[f"{s:.1f}%" for s in success_rates],
                    textposition='outside'
                ),
                row=2, col=1
            )
        
        # 4. Run distribution
        run_counts = []
        model_names = []
        
        for model_key in list(self.rankings.keys())[:10]:
            stats = self.rankings[model_key]
            run_counts.append(stats["total_runs"])
            
            from app.utils.helpers import MODEL_REGISTRY
            model_meta = MODEL_REGISTRY.get(model_key, {})
            model_names.append(model_meta.get("label", model_key))
        
        if run_counts:
            fig.add_trace(
                go.Bar(
                    x=model_names,
                    y=run_counts,
                    name="Total Runs",
                    marker_color='orange'
                ),
                row=2, col=2
            )
        
        # Update layout
        fig.update_layout(
            title="Model Performance Report",
            height=800,
            showlegend=True,
            plot_bgcolor='white'
        )
        
        # Save to HTML
        report_path = self.log_dir / output_file
        fig.write_html(str(report_path))
        
        return str(report_path)
    
    def cleanup_old_records(self, days_to_keep: int = 90):
        """Clean up records older than specified days."""
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        
        for model_key in list(self.records.keys()):
            # Filter out old records
            self.records[model_key] = [
                r for r in self.records[model_key]
                if datetime.fromisoformat(r.timestamp) > cutoff_date
            ]
            
            # Remove model if no records left
            if not self.records[model_key]:
                del self.records[model_key]
        
        # Save updated records
        self._save_records()
        
        print(f"✅ Cleaned up records older than {days_to_keep} days")