# app/components/batch_forecast.py
"""
Batch Forecasting Module
Handles forecasting for multiple items/stores in batch mode
"""
import streamlit as st # type: ignore
import traceback
from app.bootstrap import *  # ensures project root is on sys.path  # noqa: F403
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import warnings  # noqa: F401
import concurrent.futures
from pathlib import Path
import json
from datetime import datetime



class BatchForecaster:
    """Handles batch forecasting for multiple items/stores."""
    
    def __init__(self, 
                 forecast_engine_class,
                 historical_data_path: str,
                 output_dir: str = "data/batch_forecasts"):
        """
        Initialize batch forecaster.
        
        Args:
            forecast_engine_class: ForecastEngine class
            historical_data_path: Path to historical data directory or pattern
            output_dir: Directory to save batch forecast results
        """
        self.forecast_engine_class = forecast_engine_class
        self.historical_data_path = Path(historical_data_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Cache for loaded data
        self.data_cache = {}
    
    def load_item_store_data(self, item_id: str, store_id: str) -> Optional[pd.DataFrame]:
        """
        Load historical data for a specific item-store combination.
        
        Args:
            item_id: Item identifier
            store_id: Store identifier
            
        Returns:
            Historical DataFrame or None if not found
        """
        cache_key = f"{item_id}_{store_id}"
        
        if cache_key in self.data_cache:
            return self.data_cache[cache_key].copy()
        
        # Try different file patterns
        patterns = [
            f"*_ITEM-{item_id}__STORE-{store_id}.csv",
            f"*ITEM{item_id}*STORE{store_id}*.csv",
            f"*{item_id}*{store_id}*.csv"
        ]
        
        for pattern in patterns:
            files = list(self.historical_data_path.glob(pattern))
            if files:
                try:
                    df = pd.read_csv(files[0], parse_dates=["date"])
                    self.data_cache[cache_key] = df
                    return df.copy()
                except Exception as e:
                    print(f"Error loading {files[0]}: {e}")
        
        # If not found, try to find in combined data
        if self.historical_data_path.is_file():
            try:
                df = pd.read_csv(self.historical_data_path, parse_dates=["date"])
                # Filter for item and store
                if "item_nbr" in df.columns and "store_nbr" in df.columns:
                    filtered_df = df[(df["item_nbr"] == int(item_id)) & 
                                    (df["store_nbr"] == int(store_id))]
                    if not filtered_df.empty:
                        self.data_cache[cache_key] = filtered_df
                        return filtered_df.copy()
            except Exception as e:
                print(f"Error loading combined data: {e}")
        
        return None
    
    def forecast_single_item_store(self,
                                 item_id: str,
                                 store_id: str,
                                 config: Dict[str, Any],
                                 model_key: str,
                                 force_reload: bool = False) -> Dict[str, Any]:
        """
        Generate forecast for a single item-store combination.
        
        Args:
            item_id: Item identifier
            store_id: Store identifier
            config: Forecast configuration
            model_key: Model to use
            force_reload: Force reload model
            
        Returns:
            Dictionary with forecast results and metadata
        """
        try:
            # Load historical data
            historical_df = self.load_item_store_data(item_id, store_id)
            
            if historical_df is None or historical_df.empty:
                return {
                    "item_id": item_id,
                    "store_id": store_id,
                    "status": "failed",
                    "error": f"No historical data found for item {item_id} at store {store_id}"
                }
            
            # Load model
            from app.utils.helpers import load_registered_model, MODEL_REGISTRY
            
            model = load_registered_model(model_key)
            if model is None:
                return {
                    "item_id": item_id,
                    "store_id": store_id,
                    "status": "failed",
                    "error": f"Failed to load model: {model_key}"
                }
            
            model_meta = MODEL_REGISTRY[model_key]
            model_type = model_meta["model_type"]
            
            # Create feature builder
            from app.components.feature_forecast_builder import FeatureForecastBuilder
            builder = FeatureForecastBuilder(historical_df, model_type=model_type)
            
            # Build future features
            future_df = builder.build_future_features(
                start_date=pd.to_datetime(config["start_date"]),
                horizon=config["horizon"],
                frequency=config["freq"],
                onpromotion=config["onpromotion"],
                model_type=model_type
            )
            
            # Initialize forecast engine
            engine = self.forecast_engine_class(model, model_type, builder)
            
            # Generate predictions
            predictions = engine.predict(future_df, config["horizon"], config["freq"])
            
            # Get confidence intervals if available
            confidence_intervals = None
            if config.get("show_confidence", True):
                confidence_intervals = engine.get_confidence_intervals(
                    future_df,
                    confidence_level=config.get("confidence_level", 0.95)
                )
            
            # Create result DataFrame
            result_df = pd.DataFrame({
                "item_id": item_id,
                "store_id": store_id,
                "date": future_df["date"],
                "point_forecast": predictions
            })
            
            # Add confidence intervals if available
            if confidence_intervals is not None:
                lower, upper = confidence_intervals
                result_df["lower_bound"] = lower
                result_df["upper_bound"] = upper
            
            # Calculate summary statistics
            summary_stats = {
                "total_forecast": float(result_df["point_forecast"].sum()),
                "average_forecast": float(result_df["point_forecast"].mean()),
                "std_forecast": float(result_df["point_forecast"].std()),
                "min_forecast": float(result_df["point_forecast"].min()),
                "max_forecast": float(result_df["point_forecast"].max()),
                "historical_average": float(historical_df["unit_sales"].mean()),
                "growth_vs_history": ((result_df["point_forecast"].mean() / 
                                      historical_df["unit_sales"].mean() - 1) * 100 
                                     if historical_df["unit_sales"].mean() != 0 else 0)
            }
            
            # Save forecast results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"forecast_{item_id}_{store_id}_{model_key}_{timestamp}.csv"
            filepath = self.output_dir / filename
            
            result_df.to_csv(filepath, index=False)
            
            return {
                "item_id": item_id,
                "store_id": store_id,
                "status": "success",
                "model_key": model_key,
                "model_label": model_meta["label"],
                "forecast_file": str(filepath),
                "summary_stats": summary_stats,
                "forecast_dates": result_df["date"].tolist(),
                "forecast_values": result_df["point_forecast"].tolist(),
                "has_confidence": confidence_intervals is not None,
                "record_count": len(result_df)
            }
            
        except Exception as e:
            return {
                "item_id": item_id,
                "store_id": store_id,
                "status": "failed",
                "error": str(e),
                "traceback": traceback.format_exc() if "traceback" in locals() else ""
            }
    
    def forecast_batch(self,
                      item_store_pairs: List[Tuple[str, str]],
                      config: Dict[str, Any],
                      model_key: str,
                      parallel: bool = True,
                      max_workers: int = 4,
                      progress_callback=None) -> Dict[str, Dict[str, Any]]:
        """
        Generate forecasts for multiple item-store combinations.
        
        Args:
            item_store_pairs: List of (item_id, store_id) tuples
            config: Forecast configuration
            model_key: Model to use
            parallel: Whether to run in parallel
            max_workers: Maximum parallel workers
            progress_callback: Callback for progress updates
            
        Returns:
            Dictionary of forecast results keyed by "item_store"
        """
        results = {}
        total_pairs = len(item_store_pairs)
        
        st.info(f"🔄 Starting batch forecast for {total_pairs} item-store combinations...")
        
        if parallel and total_pairs > 1:
            # Parallel execution
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all tasks
                future_to_pair = {
                    executor.submit(
                        self.forecast_single_item_store,
                        item_id, store_id, config, model_key
                    ): (item_id, store_id)
                    for item_id, store_id in item_store_pairs
                }
                
                # Process results as they complete
                completed = 0
                for future in concurrent.futures.as_completed(future_to_pair):
                    item_id, store_id = future_to_pair[future]
                    key = f"{item_id}_{store_id}"
                    
                    try:
                        result = future.result()
                        results[key] = result
                    except Exception as e:
                        results[key] = {
                            "item_id": item_id,
                            "store_id": store_id,
                            "status": "failed",
                            "error": str(e)
                        }
                    
                    completed += 1
                    
                    # Update progress
                    if progress_callback:
                        progress_callback(completed, total_pairs)
                    
                    # Update Streamlit progress
                    if st.session_state.get('batch_in_progress', False):
                        st.session_state['batch_progress'] = completed / total_pairs
        
        else:
            # Sequential execution
            for i, (item_id, store_id) in enumerate(item_store_pairs, 1):
                key = f"{item_id}_{store_id}"
                
                # Update progress
                if progress_callback:
                    progress_callback(i, total_pairs)
                
                if st.session_state.get('batch_in_progress', False):
                    st.session_state['batch_progress'] = i / total_pairs
                
                # Generate forecast
                result = self.forecast_single_item_store(item_id, store_id, config, model_key)
                results[key] = result
        
        # Generate batch summary
        batch_summary = self._generate_batch_summary(results)
        
        # Save batch results
        self._save_batch_results(results, config, model_key, batch_summary)
        
        return results
    
    def _generate_batch_summary(self, results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Generate summary statistics for batch forecast."""
        
        successful = [r for r in results.values() if r.get("status") == "success"]
        failed = [r for r in results.values() if r.get("status") == "failed"]
        
        summary = {
            "total_items": len(results),
            "successful": len(successful),
            "failed": len(failed),
            "success_rate": (len(successful) / len(results) * 100) if results else 0,
            "total_forecast": sum(r.get("summary_stats", {}).get("total_forecast", 0) 
                                 for r in successful),
            "avg_growth": np.mean([r.get("summary_stats", {}).get("growth_vs_history", 0) 
                                  for r in successful]) if successful else 0,
            "timestamp": datetime.now().isoformat()
        }
        
        # Top performers by growth
        if successful:
            sorted_by_growth = sorted(
                successful,
                key=lambda x: x.get("summary_stats", {}).get("growth_vs_history", 0),
                reverse=True
            )[:5]
            
            summary["top_growth_items"] = [
                {
                    "item_id": r["item_id"],
                    "store_id": r["store_id"],
                    "growth": r.get("summary_stats", {}).get("growth_vs_history", 0),
                    "total_forecast": r.get("summary_stats", {}).get("total_forecast", 0)
                }
                for r in sorted_by_growth
            ]
        
        # Common errors
        if failed:
            error_counts = {}
            for result in failed:
                error = result.get("error", "Unknown error")
                error_counts[error] = error_counts.get(error, 0) + 1
            
            summary["common_errors"] = [
                {"error": error, "count": count}
                for error, count in sorted(error_counts.items(), 
                                          key=lambda x: x[1], reverse=True)[:5]
            ]
        
        return summary
    
    def _save_batch_results(self, 
                          results: Dict[str, Dict[str, Any]],
                          config: Dict[str, Any],
                          model_key: str,
                          summary: Dict[str, Any]):
        """Save batch forecast results to files."""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        batch_id = f"batch_{timestamp}"
        batch_dir = self.output_dir / batch_id
        batch_dir.mkdir(exist_ok=True)
        
        # Save individual results
        for key, result in results.items():
            if result.get("status") == "success" and "forecast_file" in result:
                # Move file to batch directory
                old_path = Path(result["forecast_file"])
                new_path = batch_dir / old_path.name
                old_path.rename(new_path)
                result["forecast_file"] = str(new_path)
        
        # Save results metadata
        metadata = {
            "batch_id": batch_id,
            "timestamp": timestamp,
            "config": config,
            "model_key": model_key,
            "summary": summary,
            "results": results
        }
        
        metadata_file = batch_dir / "batch_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        # Save summary CSV
        summary_rows = []
        for key, result in results.items():
            if result.get("status") == "success":
                row = {
                    "item_id": result["item_id"],
                    "store_id": result["store_id"],
                    "total_forecast": result.get("summary_stats", {}).get("total_forecast", 0),
                    "average_forecast": result.get("summary_stats", {}).get("average_forecast", 0),
                    "growth_vs_history": result.get("summary_stats", {}).get("growth_vs_history", 0),
                    "status": "success"
                }
            else:
                row = {
                    "item_id": result["item_id"],
                    "store_id": result["store_id"],
                    "total_forecast": 0,
                    "average_forecast": 0,
                    "growth_vs_history": 0,
                    "status": "failed",
                    "error": result.get("error", "Unknown")
                }
            summary_rows.append(row)
        
        if summary_rows:
            summary_df = pd.DataFrame(summary_rows)
            summary_file = batch_dir / "batch_summary.csv"
            summary_df.to_csv(summary_file, index=False)
        
        print(f"✅ Batch forecast saved to: {batch_dir}")
    
    def load_batch_results(self, batch_id: str) -> Optional[Dict[str, Any]]:
        """Load batch forecast results from disk."""
        
        batch_dir = self.output_dir / batch_id
        
        if not batch_dir.exists():
            return None
        
        metadata_file = batch_dir / "batch_metadata.json"
        
        if metadata_file.exists():
            try:
                with open(metadata_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading batch results: {e}")
        
        return None
    
    def generate_batch_report(self, 
                            batch_id: str,
                            output_file: str = "batch_report.html") -> str:
        """Generate HTML report for batch forecast."""
        
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        
        # Load batch data
        batch_data = self.load_batch_results(batch_id)
        
        if not batch_data:
            return "Batch not found"
        
        results = batch_data.get("results", {})
        summary = batch_data.get("summary", {})
        
        # Create figure
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("Forecast Distribution", 
                          "Success vs Failure",
                          "Growth Distribution", 
                          "Top Performers"),
            vertical_spacing=0.15,
            horizontal_spacing=0.15
        )
        
        successful = [r for r in results.values() if r.get("status") == "success"]
        
        # 1. Forecast distribution
        if successful:
            forecast_totals = [r.get("summary_stats", {}).get("total_forecast", 0) 
                              for r in successful]
            
            fig.add_trace(
                go.Histogram(
                    x=forecast_totals,
                    name="Forecast Totals",
                    nbinsx=20,
                    marker_color='lightblue'
                ),
                row=1, col=1
            )
        
        # 2. Success vs Failure pie chart
        success_count = len(successful)
        failure_count = len(results) - success_count
        
        fig.add_trace(
            go.Pie(
                labels=["Successful", "Failed"],
                values=[success_count, failure_count],
                name="Success Rate",
                marker_colors=['green', 'red']
            ),
            row=1, col=2
        )
        
        # 3. Growth distribution
        if successful:
            growth_rates = [r.get("summary_stats", {}).get("growth_vs_history", 0) 
                           for r in successful]
            
            fig.add_trace(
                go.Box(
                    y=growth_rates,
                    name="Growth Rates",
                    boxpoints='all',
                    marker_color='orange'
                ),
                row=2, col=1
            )
        
        # 4. Top performers bar chart
        top_items = summary.get("top_growth_items", [])[:10]
        if top_items:
            item_labels = [f"{item['item_id']}_{item['store_id']}" 
                          for item in top_items]
            growth_values = [item['growth'] for item in top_items]
            
            fig.add_trace(
                go.Bar(
                    x=item_labels,
                    y=growth_values,
                    name="Top Growth",
                    marker_color='lightgreen',
                    text=[f"{g:.1f}%" for g in growth_values],
                    textposition='outside'
                ),
                row=2, col=2
            )
        
        # Update layout
        fig.update_layout(
            title=f"Batch Forecast Report: {batch_id}",
            height=800,
            showlegend=False,
            plot_bgcolor='white'
        )
        
        # Save to HTML
        report_path = self.output_dir / batch_id / output_file
        fig.write_html(str(report_path))
        
        return str(report_path)
    
    def get_available_batches(self) -> List[Dict[str, Any]]:
        """Get list of available batch forecasts."""
        
        batches = []
        
        for batch_dir in self.output_dir.iterdir():
            if batch_dir.is_dir() and batch_dir.name.startswith("batch_"):
                metadata_file = batch_dir / "batch_metadata.json"
                
                if metadata_file.exists():
                    try:
                        with open(metadata_file, 'r') as f:
                            metadata = json.load(f)
                        
                        batches.append({
                            "batch_id": batch_dir.name,
                            "timestamp": metadata.get("timestamp", ""),
                            "item_count": len(metadata.get("results", {})),
                            "success_rate": metadata.get("summary", {}).get("success_rate", 0),
                            "model_key": metadata.get("model_key", "")
                        })
                    except:  # noqa: E722
                        continue
        
        # Sort by timestamp (newest first)
        batches.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        
        return batches