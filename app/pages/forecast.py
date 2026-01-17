# app/pages/forecast.py 

from typing import Dict, Any
from app.bootstrap import *  # ensures project root is on sys.path  # noqa: F403

import streamlit as st # type: ignore
import pandas as pd
import numpy as np  # noqa: F401
from pathlib import Path
import plotly.graph_objects as go

from app.utils import (
    MODEL_REGISTRY,
    load_registered_model,
    
)

from app.ui.forecast_ui import ForecastUI
from app.ui.forecast_engine import ForecastEngine
from app.utils.visualizer import ForecastVisualizer

from app.components.performance_tracker import ModelPerformanceTracker
from app.components.feature_forecast_builder import FeatureForecastBuilder
from src import get_path



"""
Main Forecast Application with All Features
Enhanced with Auto-Best Model Selection, Executive Dashboard, Performance Tracking, and Batch Forecasting
"""




class Forecast:
    """Main forecast application."""
    
    def __init__(self):
        """Initialize forecast application."""
        self.historical_df = None
        self.model = None
        self.model_type = None
        self.meta = None
        self.builder = None
        self.visualizer = None
        self.ui = ForecastUI()
        
        # Initialize new components
        self.performance_tracker = None
        self.model_selector = None
        self.batch_forecaster = None
        
        # Initialize session state for batch forecasting
        if 'batch_in_progress' not in st.session_state:
            st.session_state.batch_in_progress = False
            st.session_state.batch_progress = 0.0
    
    @staticmethod
    @st.cache_data
    def load_historical() -> pd.DataFrame:
        """Load historical data from disk."""
        filename = "final_train_dataset__MAXDATE-2016-12-31__STORE-24__ITEM-105577.csv"
        path = Path(get_path("filtered")) / filename

        if not path.exists():
            st.error(f"❌ Historical data file not found: {path}")
            st.info("Please ensure the data file exists in the correct location.")
            st.stop()

        df = pd.read_csv(path, parse_dates=["date"])
        return df
    
    def setup_sidebar(self) -> Dict[str, Any]:
        """Setup sidebar configuration interface."""
        st.sidebar.header("🔧 Forecast Settings")

        # Model selection
        model_key = st.sidebar.selectbox(
            "Choose Model",
            options=list(MODEL_REGISTRY.keys()),
            format_func=lambda k: MODEL_REGISTRY[k]["label"],
            help="Select a trained model for forecasting"
        )

        # Display model info
        selected_meta = MODEL_REGISTRY[model_key]
        st.sidebar.info(f"""
        **Model Type:** {selected_meta['model_type'].upper()}  
        **Week:** Week {selected_meta['week']}  
        """)

        # Frequency selection
        frequency_options = {
            "D": "Daily",
            "W": "Weekly",
            "M": "Monthly",
            "Q": "Quarterly",
            "Y": "Yearly"
        }
        freq = st.sidebar.selectbox(
            "Frequency",
            options=list(frequency_options.keys()),
            format_func=lambda k: frequency_options[k],
            help="Forecast frequency"
        )

        # Horizon with context-aware limits
        horizon_options = {
            "D": (7, 365, 30),
            "W": (4, 52, 12),
            "M": (3, 36, 12),
            "Q": (2, 16, 8),
            "Y": (1, 10, 3)
        }
        min_h, max_h, default_h = horizon_options.get(freq, (7, 365, 30))
        horizon = st.sidebar.slider(
            f"Forecast Horizon ({frequency_options[freq].lower()})",
            min_h, max_h, default_h,
            help=f"Number of {frequency_options[freq].lower()} periods to forecast"
        )

        # Start date
        if self.historical_df is not None and not self.historical_df.empty:
            max_historical_date = self.historical_df['date'].max()
            default_start = max_historical_date + pd.Timedelta(days=1)
        else:
            default_start = pd.Timestamp.today()

        start_date = st.sidebar.date_input(
            "Start Date",
            value=default_start,
            help="Start date for the forecast"
        )

        # Promotion
        onpromotion = st.sidebar.selectbox(
            "On Promotion?",
            options=[0, 1],
            format_func=lambda x: "Yes ✅" if x == 1 else "No ❌",
            help="Whether items are on promotion during forecast period"
        )

        # Confidence level
        confidence_level = st.sidebar.slider(
            "Confidence Level",
            min_value=0.50,
            max_value=0.99,
            value=0.95,
            step=0.01,
            help="Select the confidence interval level for forecast uncertainty"
        )

        # Visualization options
        st.sidebar.markdown("---")
        st.sidebar.subheader("📊 Visualization Options")

        plot_types = st.sidebar.multiselect(
            "Select Plots",
            options=[
                "Timeseries",
                "Distribution",
                "Uncertainty",
                "Components",
                "Heatmap",
                "Waterfall"
            ],
            default=["Timeseries", "Distribution", "Uncertainty"],
            help="Choose which visualizations to display"
        )

        show_confidence = st.sidebar.checkbox(
            "Show Confidence Intervals",
            value=True,
            help="Display confidence intervals if available"
        )
        download_plots = st.sidebar.checkbox(
            "Enable Plot Download",
            value=True,
            help="Allow downloading plots as PNG files"
        )

        return {
            "model_key": model_key,
            "freq": freq,
            "horizon": horizon,
            "start_date": start_date,
            "onpromotion": onpromotion,
            "plot_types": plot_types,
            "show_confidence": show_confidence,
            "download_plots": download_plots,
            "confidence_level": confidence_level
        }
    
    def setup_dynamic_filters(self) -> Dict[str, Any]:
        """Setup dynamic filters that update dashboard in real-time."""
        st.sidebar.markdown("---")
        st.sidebar.subheader("🎯 Dynamic Dashboard Filters")
        
        filters = {}
        
        # Time-based filtering
        filters["time_focus"] = st.sidebar.select_slider(
            "Forecast Period Focus",
            options=["All Periods", "First Week", "First Month", 
                    "First Quarter", "Key Periods Only", "Peak Periods"],
            value="All Periods",
            help="Focus analysis on specific time periods"
        )
        
        # Value threshold for highlighting
        if hasattr(self, 'historical_df') and not self.historical_df.empty:
            default_threshold = self.historical_df["unit_sales"].quantile(0.75)
        else:
            default_threshold = 100
        
        filters["highlight_threshold"] = st.sidebar.number_input(
            "Highlight Values Above",
            min_value=0.0,
            value=float(default_threshold),
            step=10.0,
            help="Values above this threshold will be highlighted"
        )
        
        # Analysis depth
        filters["analysis_depth"] = st.sidebar.select_slider(
            "Analysis Depth",
            options=["Basic", "Standard", "Detailed", "Comprehensive"],
            value="Standard",
            help="Level of detail in analysis"
        )
        
        # Trend analysis options
        st.sidebar.markdown("#### 📈 Trend Analysis")
        
        filters["show_trend_line"] = st.sidebar.checkbox(
            "Show Trend Line", 
            value=True,
            help="Display linear trend line"
        )
        
        filters["show_moving_average"] = st.sidebar.checkbox(
            "Show Moving Average",
            value=True,
            help="Display 7-day moving average"
        )
        
        filters["decompose_seasonality"] = st.sidebar.checkbox(
            "Decompose Seasonality",
            value=False,
            help="Break down into trend, seasonal, and residual components"
        )
        
        # Comparison options
        st.sidebar.markdown("#### 🔄 Comparison Options")
        
        filters["compare_with"] = st.sidebar.multiselect(
            "Compare With",
            options=["Historical Average", "Previous Month", 
                    "Same Period Last Year", "Industry Benchmark",
                    "Best Case Scenario", "Worst Case Scenario"],
            default=["Historical Average"],
            help="Select comparisons to display"
        )
        
        # Visualization style
        st.sidebar.markdown("#### 🎨 Visualization Style")
        
        filters["chart_style"] = st.sidebar.selectbox(
            "Chart Style",
            options=["Professional", "Minimal", "Colorful", "Dark Mode"],
            index=0,
            help="Select visualization style"
        )
        
        filters["animation_enabled"] = st.sidebar.checkbox(
            "Enable Animations",
            value=True,
            help="Add animations to charts"
        )
        
        # Advanced filters (collapsible)
        with st.sidebar.expander("⚙️ Advanced Filters"):
            filters["smoothing_factor"] = st.slider(
                "Smoothing Factor",
                min_value=0.0,
                max_value=1.0,
                value=0.3,
                step=0.1,
                help="Apply smoothing to reduce noise"
            )
            
            filters["outlier_threshold"] = st.slider(
                "Outlier Detection Threshold",
                min_value=1.0,
                max_value=5.0,
                value=3.0,
                step=0.5,
                help="Z-score threshold for outlier detection"
            )
            
            filters["confidence_band"] = st.select_slider(
                "Confidence Band Display",
                options=["None", "Light", "Standard", "Emphasized"],
                value="Standard"
            )
        
        return filters
    
    def run_forecast_pipeline(self, config: Dict[str, Any]):
        """Execute the complete forecast pipeline."""
        try:
            # Step 1: Load model
            st.info("🔄 Loading model...")
            self.meta = MODEL_REGISTRY[config["model_key"]]
            self.model_type = self.meta["model_type"]
            self.model = load_registered_model(config["model_key"])

            if self.model is None:
                self.ui.display_error_message(
                    ValueError("Model could not be loaded"),
                    "during model loading"
                )
                return

            st.success(f"✅ Loaded model: **{self.meta['label']}**")

            # Step 2: Build features
            st.info("🔄 Building future features...")
            self.builder = FeatureForecastBuilder(
                self.historical_df,
                model_type=self.model_type
            )

            future_df = self.builder.build_future_features(
                start_date=pd.to_datetime(config["start_date"]),
                horizon=config["horizon"],
                frequency=config["freq"],
                onpromotion=config["onpromotion"],
                model_type=self.model_type
            )
            
           

            # Ensure month column for heatmap
            if "month" not in future_df.columns and "date" in future_df.columns:
                future_df["month"] = pd.to_datetime(future_df["date"]).dt.month

            st.success(f"✅ Built {len(future_df)} feature rows")

            # Display feature statistics
            with st.expander("📋 Feature Statistics", expanded=False):
                self.ui.display_feature_statistics(self.builder)

            # Step 3: Generate forecast
            st.info("🔄 Generating forecast...")
            engine = ForecastEngine(self.model, self.model_type, self.builder)
            preds = engine.predict(future_df, config["horizon"], config["freq"])

            st.success(f"✅ Generated {len(preds)} predictions")
            
            

            # Step 4: Create results DataFrame
            if self.model_type.lower() == "prophet":
                if "ds" not in future_df.columns:
                    raise ValueError("Prophet model requires 'ds' column but it was not found.")
                date_col = pd.to_datetime(future_df["ds"])
            else:
                if "date" not in future_df.columns:
                    raise ValueError("Non-Prophet model requires 'date' column but it was not found.")
                date_col = pd.to_datetime(future_df["date"])

            result_df = pd.DataFrame({
                "date": date_col,
                "point_forecast": preds,
            })
            
            st.session_state["last_forecast_df"] = result_df
            st.session_state["last_config"] = config


            # Add month column if missing
            if "month" not in result_df.columns:
                result_df["month"] = pd.to_datetime(result_df["date"]).dt.month

            # Step 5: Try to get confidence intervals
            if config["show_confidence"]:
                st.info("🔄 Calculating confidence intervals...")
                try:
                    confidence_intervals = engine.get_confidence_intervals(
                        future_df,
                        confidence_level=config["confidence_level"]
                    )
                except TypeError:
                    # Backward compatibility if engine doesn't accept confidence_level yet
                    confidence_intervals = engine.get_confidence_intervals(future_df)

                if confidence_intervals is not None:
                    lower, upper = confidence_intervals
                    result_df["lower_bound"] = lower
                    result_df["upper_bound"] = upper
                    st.success(f"✅ Confidence intervals calculated at {config['confidence_level']*100:.0f}%")
                else:
                    st.info("ℹ️ Confidence intervals not available for this model")

            # Step 6: Display Executive Dashboard if enabled
            if config.get("enable_executive_dashboard", True):
                from app.components.executive_dashboard import ExecutiveDashboard
                ExecutiveDashboard.create_kpi_dashboard(
                    result_df, 
                    self.historical_df,
                    config
                )
            
            # Step 7: Display results summary
            st.markdown("---")
            self.ui.display_forecast_summary(result_df)

            # Step 8: Initialize visualizer
            self.visualizer = ForecastVisualizer(self.historical_df)

            # Step 9: Apply dynamic filters if enabled
            if "dynamic_filters" in config:
                st.info("🎯 Applying dynamic filters...")
                filtered_df = self.apply_dynamic_filters(result_df, config["dynamic_filters"])
                
                # Create filtered visualizations
                filtered_figures = self.create_filtered_visualizations(
                    filtered_df, 
                    config["dynamic_filters"], 
                    self.visualizer
                )
                
                # Display filtered figures
                for fig_name, fig in filtered_figures.items():
                    st.subheader(f"📈 {fig_name.replace('_', ' ').title()}")
                    st.plotly_chart(fig, use_container_width=True)
            
            else:
                # Step 10: Create standard plots
                st.markdown("---")
                st.subheader("📊 Visualizations")

                if config["plot_types"]:
                    self.ui.create_plots(
                        self.visualizer,
                        result_df,
                        config["plot_types"],
                        self.meta,
                        config["show_confidence"],
                        config["download_plots"],
                        self.historical_df,
                        actual_df=None,
                        confidence_level=config["confidence_level"]
                    )
                else:
                    st.info("No visualizations selected. Choose plots from the sidebar.")

            # Step 11: Display data table
            st.markdown("---")
            self.ui.display_data_table(
                result_df,
                self.meta,
                config["freq"],
                config["horizon"]
            )

            # Step 12: Log forecast run for performance tracking
            if self.performance_tracker:
                try:
                    forecast_id = self.performance_tracker.log_forecast_run(
                        model_key=config["model_key"],
                        config=config,
                        forecast_df=result_df,
                        actual_df=None,  # No actual data for future forecasts
                        notes=f"Forecast run via {self.meta['label']}"
                    )
                    st.info(f"📊 Forecast logged with ID: {forecast_id}")
                except Exception as e:
                    st.warning(f"Could not log forecast run: {e}")

            # Success message
            st.markdown("---")
            self.ui.display_success_box(
                f"""
                Forecast completed successfully!

                - **Model:** {self.meta['label']}
                - **Horizon:** {config['horizon']} {config['freq']}
                - **Start Date:** {config['start_date']}
                - **Predictions:** {len(result_df)} periods
                - **Confidence Level:** {config['confidence_level']*100:.0f}%
                """,
                title="✅ Forecast Complete"
            )

        except Exception as e:
            self.ui.display_error_message(e, "during forecasting")

            # Additional debugging info
            if st.checkbox("Show debugging information"):
                st.write("**Configuration:**")
                st.json(config)

                st.write("**Model Info:**")
                st.write(f"- Type: {self.model_type}")
                st.write(f"- Model object: {type(self.model) if self.model else 'None'}")

                if self.builder:
                    st.write("**Builder Info:**")
                    st.write(f"- Historical data shape: {self.builder.historical_df.shape}")
    
    def apply_dynamic_filters(self, forecast_df: pd.DataFrame, filters: Dict[str, Any]) -> pd.DataFrame:
        """
        Apply dynamic filters to forecast data.
        
        Args:
            forecast_df: Original forecast DataFrame
            filters: Dictionary of filter settings
            
        Returns:
            Filtered DataFrame
        """
        filtered_df = forecast_df.copy()
        
        # Apply time-based filtering
        if filters["time_focus"] != "All Periods":
            total_days = len(filtered_df)  # noqa: F841
            
            if filters["time_focus"] == "First Week":
                filtered_df = filtered_df.head(7)
            elif filters["time_focus"] == "First Month":
                filtered_df = filtered_df.head(30)
            elif filters["time_focus"] == "First Quarter":
                filtered_df = filtered_df.head(90)
            elif filters["time_focus"] == "Key Periods Only":
                # Keep only periods with significant changes
                pct_change = filtered_df["point_forecast"].pct_change().abs()
                filtered_df = filtered_df[pct_change > pct_change.quantile(0.75)]
            elif filters["time_focus"] == "Peak Periods":
                # Keep top 20% of values
                threshold = filtered_df["point_forecast"].quantile(0.8)
                filtered_df = filtered_df[filtered_df["point_forecast"] >= threshold]
        
        # Add highlight column for values above threshold
        filtered_df["is_highlighted"] = filtered_df["point_forecast"] > filters["highlight_threshold"]
        
        # Apply smoothing if requested
        if filters["smoothing_factor"] > 0:
            filtered_df["point_forecast_smoothed"] = filtered_df["point_forecast"].ewm(
                alpha=filters["smoothing_factor"]
            ).mean()
        
        # Add trend components if decomposition requested
        if filters["decompose_seasonality"] and len(filtered_df) >= 14:
            try:
                from statsmodels.tsa.seasonal import seasonal_decompose
                
                # Ensure we have enough data
                if len(filtered_df) >= 30:
                    # Create time series for decomposition
                    ts = filtered_df.set_index("date")["point_forecast"]
                    
                    # Use appropriate period based on data frequency
                    period = 7  # Weekly seasonality for daily data
                    
                    decomposition = seasonal_decompose(ts, model='additive', period=period)
                    
                    # Add components to DataFrame
                    filtered_df["trend"] = decomposition.trend.values
                    filtered_df["seasonal"] = decomposition.seasonal.values
                    filtered_df["residual"] = decomposition.resid.values
            except Exception as e:
                st.info(f"Seasonal decomposition not available: {e}")
        
        return filtered_df
    
    def create_filtered_visualizations(self, 
                                     forecast_df: pd.DataFrame,
                                     filters: Dict[str, Any],
                                     visualizer: ForecastVisualizer) -> Dict[str, go.Figure]:
        """
        Create visualizations with applied filters.
        
        Args:
            forecast_df: Filtered forecast DataFrame
            filters: Filter settings
            visualizer: ForecastVisualizer instance
            
        Returns:
            Dictionary of Plotly figures
        """
        figures = {}
        
        # Determine chart colors based on style
        color_schemes = {
            "Professional": {"primary": "#2E86AB", "accent": "#A23B72", "background": "#F8F9FA"},
            "Minimal": {"primary": "#333333", "accent": "#666666", "background": "#FFFFFF"},
            "Colorful": {"primary": "#FF6B6B", "accent": "#4ECDC4", "background": "#FFEAA7"},
            "Dark Mode": {"primary": "#00D4AA", "accent": "#FF6B8B", "background": "#1A1A2E"}
        }
        
        colors = color_schemes.get(filters["chart_style"], color_schemes["Professional"])
        
        # 1. Main forecast plot with filters
        fig_main = go.Figure()
        
        # Add forecast line
        line_color = colors["primary"]
        if filters["highlight_threshold"] > 0:
            # Color points above threshold differently
            above_threshold = forecast_df["point_forecast"] > filters["highlight_threshold"]
            
            fig_main.add_trace(go.Scatter(
                x=forecast_df.loc[above_threshold, "date"],
                y=forecast_df.loc[above_threshold, "point_forecast"],
                mode="markers",
                name="Above Threshold",
                marker=dict(color="red", size=8, symbol="star")
            ))
            
            fig_main.add_trace(go.Scatter(
                x=forecast_df.loc[~above_threshold, "date"],
                y=forecast_df.loc[~above_threshold, "point_forecast"],
                mode="lines+markers",
                name="Forecast",
                line=dict(color=line_color, width=2),
                marker=dict(size=4)
            ))
        else:
            fig_main.add_trace(go.Scatter(
                x=forecast_df["date"],
                y=forecast_df["point_forecast"],
                mode="lines+markers",
                name="Forecast",
                line=dict(color=line_color, width=2),
                marker=dict(size=4)
            ))
        
        # Add trend line if requested
        if filters["show_trend_line"]:
            x_numeric = np.arange(len(forecast_df))
            y = forecast_df["point_forecast"].values
            
            # Calculate linear trend
            z = np.polyfit(x_numeric, y, 1)
            p = np.poly1d(z)
            trend_line = p(x_numeric)
            
            fig_main.add_trace(go.Scatter(
                x=forecast_df["date"],
                y=trend_line,
                mode="lines",
                name="Trend Line",
                line=dict(color="red", width=2, dash="dash")
            ))
        
        # Add moving average if requested
        if filters["show_moving_average"]:
            ma_window = min(7, len(forecast_df))
            moving_avg = forecast_df["point_forecast"].rolling(window=ma_window).mean()
            
            fig_main.add_trace(go.Scatter(
                x=forecast_df["date"],
                y=moving_avg,
                mode="lines",
                name=f"{ma_window}-Day Moving Avg",
                line=dict(color="green", width=2, dash="dot")
            ))
        
        # Update layout
        fig_main.update_layout(
            title="Filtered Forecast View",
            xaxis_title="Date",
            yaxis_title="Unit Sales",
            hovermode="x unified",
            plot_bgcolor=colors["background"],
            height=400,
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
        )
        
        figures["filtered_forecast"] = fig_main
        
        # 2. Comparison plot if requested
        if filters["compare_with"] and hasattr(self, 'historical_df'):
            fig_comparison = self._create_comparison_plot(forecast_df, filters)
            figures["comparison"] = fig_comparison
        
        # 3. Decomposition plot if requested
        if filters["decompose_seasonality"] and all(col in forecast_df.columns 
                                                  for col in ["trend", "seasonal", "residual"]):
            fig_decomposition = self._create_decomposition_plot(forecast_df)
            figures["decomposition"] = fig_decomposition
        
        # 4. Highlight distribution plot
        fig_dist = go.Figure()
        
        if "is_highlighted" in forecast_df.columns:
            # Create histogram with highlighted values
            all_values = forecast_df["point_forecast"]  # noqa: F841
            highlighted = forecast_df.loc[forecast_df["is_highlighted"], "point_forecast"]
            normal = forecast_df.loc[~forecast_df["is_highlighted"], "point_forecast"]
            
            fig_dist.add_trace(go.Histogram(
                x=normal,
                name="Normal Values",
                marker_color=colors["primary"],
                opacity=0.7
            ))
            
            fig_dist.add_trace(go.Histogram(
                x=highlighted,
                name="Highlighted (Above Threshold)",
                marker_color="red",
                opacity=0.7
            ))
            
            fig_dist.update_layout(
                title="Value Distribution with Highlights",
                xaxis_title="Forecast Value",
                yaxis_title="Frequency",
                barmode="overlay",
                height=300
            )
            
            figures["highlight_distribution"] = fig_dist
        
        return figures
    
    def _create_comparison_plot(self, forecast_df: pd.DataFrame, filters: Dict[str, Any]) -> go.Figure:
        """Create comparison plot based on filter settings."""
        fig = go.Figure()
        
        # Add forecast
        fig.add_trace(go.Scatter(
            x=forecast_df["date"],
            y=forecast_df["point_forecast"],
            mode="lines",
            name="Current Forecast",
            line=dict(color="#2E86AB", width=3)
        ))
        
        # Add comparisons
        if "Historical Average" in filters["compare_with"]:
            hist_avg = self.historical_df["unit_sales"].mean()
            fig.add_hline(
                y=hist_avg,
                line_dash="dash",
                line_color="green",
                annotation_text=f"Historical Avg: {hist_avg:.1f}",
                annotation_position="bottom right"
            )
        
        if "Previous Month" in filters["compare_with"] and len(self.historical_df) >= 30:
            prev_month = self.historical_df["unit_sales"].tail(30).mean()
            fig.add_hline(
                y=prev_month,
                line_dash="dot",
                line_color="orange",
                annotation_text=f"Prev Month Avg: {prev_month:.1f}",
                annotation_position="bottom left"
            )
        
        if "Same Period Last Year" in filters["compare_with"] and len(self.historical_df) >= 365:
            # This is simplified - would need proper date alignment in real implementation
            same_period = self.historical_df["unit_sales"].tail(365).head(len(forecast_df)).mean()
            fig.add_hline(
                y=same_period,
                line_dash="longdash",
                line_color="purple",
                annotation_text=f"Same Period Last Year: {same_period:.1f}"
            )
        
        fig.update_layout(
            title="Forecast Comparisons",
            xaxis_title="Date",
            yaxis_title="Unit Sales",
            height=400
        )
        
        return fig
    
    def _create_decomposition_plot(self, forecast_df: pd.DataFrame) -> go.Figure:
        """Create seasonal decomposition plot."""
        from plotly.subplots import make_subplots
        
        fig = make_subplots(
            rows=4, cols=1,
            subplot_titles=["Original", "Trend", "Seasonal", "Residual"],
            vertical_spacing=0.08
        )
        
        # Original
        fig.add_trace(
            go.Scatter(
                x=forecast_df["date"],
                y=forecast_df["point_forecast"],
                mode="lines",
                name="Original",
                line=dict(color="blue", width=2)
            ),
            row=1, col=1
        )
        
        # Trend
        fig.add_trace(
            go.Scatter(
                x=forecast_df["date"],
                y=forecast_df["trend"],
                mode="lines",
                name="Trend",
                line=dict(color="red", width=2)
            ),
            row=2, col=1
        )
        
        # Seasonal
        fig.add_trace(
            go.Scatter(
                x=forecast_df["date"],
                y=forecast_df["seasonal"],
                mode="lines",
                name="Seasonal",
                line=dict(color="green", width=2)
            ),
            row=3, col=1
        )
        
        # Residual
        fig.add_trace(
            go.Scatter(
                x=forecast_df["date"],
                y=forecast_df["residual"],
                mode="markers",
                name="Residual",
                marker=dict(color="purple", size=4)
            ),
            row=4, col=1
        )
        
        fig.update_layout(
            title="Seasonal Decomposition",
            height=600,
            showlegend=False
        )
        
        return fig
    def render_visualization_tab(self):
        """Render the Visualization tab using the last forecast results."""
        st.subheader("📊 Visualizations")

        # No forecast yet
        if "last_forecast_df" not in st.session_state:
            st.info("Run a forecast first (Tab 2) to see visualizations here.")
            return

        # Load stored results
        result_df = st.session_state["last_forecast_df"]
        config = st.session_state["last_config"]

        # ---------------------------------------------------------
        # 1. Executive Dashboard (same as pipeline)
        # ---------------------------------------------------------
        from app.components.executive_dashboard import ExecutiveDashboard
        st.markdown("## 📊 Executive Dashboard")
        ExecutiveDashboard.create_kpi_dashboard(
            result_df,
            self.historical_df,
            config
        )

        # ---------------------------------------------------------
        # 2. Dynamic Filtered Visualizations (same as pipeline)
        # ---------------------------------------------------------
        st.markdown("---")
        st.subheader("🎯 Filtered Visualizations")

        filtered_df = self.apply_dynamic_filters(result_df, config["dynamic_filters"])
        filtered_figures = self.create_filtered_visualizations(
            filtered_df,
            config["dynamic_filters"],
            ForecastVisualizer(self.historical_df)
        )

        for fig_name, fig in filtered_figures.items():
            st.subheader(f"📈 {fig_name.replace('_', ' ').title()}")
            st.plotly_chart(fig, use_container_width=True)

        # ---------------------------------------------------------
        # 3. Standard Plots (same as pipeline)
        # ---------------------------------------------------------
        st.markdown("---")
        st.subheader("📊 Standard Plots")

        self.visualizer = ForecastVisualizer(self.historical_df)

        self.ui.create_plots(
            self.visualizer,
            result_df,
            config["plot_types"],
            self.meta,
            config["show_confidence"],
            config["download_plots"],
            self.historical_df,
            actual_df=None,
            confidence_level=config["confidence_level"]
        )

    def run(self):
        """Run the forecast application."""
        st.title("🔮 Advanced Forecasting Dashboard")
        st.markdown("---")
        
        # ---------------------------------------------------------
        # Main tabs (Visualization first)
        # ---------------------------------------------------------
        tab_vis, tab_single, tab_auto, tab_perf, tab_batch = st.tabs([
            "📊 Visualizations",
            "📈 Single Forecast",
            "🤖 Auto Model Selection",
            "📉 Performance Analytics",
            "🔢 Batch Forecasting"
        ])
        
        # ---------------------------------------------------------
        # Load historical data ONCE
        # ---------------------------------------------------------
        if self.historical_df is None:
            try:
                self.historical_df = self.load_historical()
                st.success(f"✅ Loaded historical data: {len(self.historical_df):,} records")

                from app.components.model_selector import AutoBestModelSelector
                self.performance_tracker = ModelPerformanceTracker()
                self.model_selector = AutoBestModelSelector(self.historical_df)

            except Exception as e:
                st.error(f"❌ Failed to load historical data: {str(e)}")
                st.stop()

        # ---------------------------------------------------------
        # Wrapper: Only for tabs that RETURN a config
        # ---------------------------------------------------------
        def _inject_pipeline_and_dashboard(run_method):
            def wrapper():
                result = run_method()

                # Only run pipeline if method returns a config
                if isinstance(result, dict) and "config" in result:
                    config = result["config"]

                    # Always enable dynamic filters
                    if "dynamic_filters" not in config:
                        config["dynamic_filters"] = self.setup_dynamic_filters()

                    # Always enable Executive Dashboard
                    config["enable_executive_dashboard"] = True

                    # Run forecast pipeline
                    result_df = self.run_forecast_pipeline(config)

                    # Store result for visualization tab
                    st.session_state["last_forecast_df"] = result_df
                    st.session_state["last_config"] = config

                    return {
                        "result_df": result_df,
                        "config": config
                    }

                return result
            return wrapper

        # ---------------------------------------------------------
        # Apply wrapper ONLY to forecast-producing tabs
        # ---------------------------------------------------------
        self._run_single_forecast = _inject_pipeline_and_dashboard(self._run_single_forecast)
        self._run_auto_model_selection = _inject_pipeline_and_dashboard(self._run_auto_model_selection)
        self._run_performance_analytics = _inject_pipeline_and_dashboard(self._run_performance_analytics)
        self._run_batch_forecasting = _inject_pipeline_and_dashboard(self._run_batch_forecasting)

        # ---------------------------------------------------------
        # TAB 1 — VISUALIZATION TAB
        # ---------------------------------------------------------
        with tab_vis:
            self.render_visualization_tab()

        # ---------------------------------------------------------
        # TAB 2 — SINGLE FORECAST
        # ---------------------------------------------------------
        with tab_single:
            self._run_single_forecast()

        # ---------------------------------------------------------
        # TAB 3 — AUTO MODEL SELECTION
        # ---------------------------------------------------------
        with tab_auto:
            self._run_auto_model_selection()

        # ---------------------------------------------------------
        # TAB 4 — PERFORMANCE ANALYTICS
        # ---------------------------------------------------------
        with tab_perf:
            self._run_performance_analytics()

        # ---------------------------------------------------------
        # TAB 5 — BATCH FORECASTING
        # ---------------------------------------------------------
        with tab_batch:
            self._run_batch_forecasting()


    
    def _run_single_forecast(self):
        """Run single forecast with all enhancements."""
        st.markdown("""
        Generate forecasts using trained models. Select your model, configure parameters,
        and visualize predictions with confidence intervals.
        """)
        
        # Display data info
        with st.expander("📊 Historical Data Info", expanded=False):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Records", f"{len(self.historical_df):,}")
            with col2:
                date_range = self.historical_df['date'].max() - self.historical_df['date'].min()
                st.metric(
                    "Date Range",
                    f"{date_range.days:,} days",
                    help=f"From {self.historical_df['date'].min().strftime('%Y-%m-%d')} "
                         f"to {self.historical_df['date'].max().strftime('%Y-%m-%d')}"
                )
            with col3:
                avg_sales = self.historical_df['unit_sales'].mean()
                st.metric("Avg Sales", f"{avg_sales:.2f}")
        
        # Setup sidebar and get configuration
        config = self.setup_sidebar()
        
        # Add dynamic filters
        st.sidebar.markdown("---")
        if st.sidebar.checkbox("🎯 Enable Dynamic Filters", value=False):
            dynamic_filters = self.setup_dynamic_filters()
            config["dynamic_filters"] = dynamic_filters
        
        # Add executive dashboard option
        st.sidebar.markdown("---")
        config["enable_executive_dashboard"] = st.sidebar.checkbox(
            "📊 Enable Executive Dashboard", 
            value=True
        )
        
        # Run forecast button
        if st.sidebar.button("🚀 Run Forecast", type="primary", use_container_width=True):
            with st.spinner(f"Generating {config['horizon']}-period forecast..."):
                self.run_forecast_pipeline(config)
        
        else:
            # Show instructions
            st.info("""
            👈 **Get Started:**

            1. Select a model from the sidebar
            2. Configure forecast parameters (horizon, frequency, dates)
            3. Choose visualization options and confidence level
            4. Click **"Run Forecast"** to generate predictions
            """)
    
    def _run_auto_model_selection(self):
        """Run auto model selection interface."""
        st.header("🤖 Auto-Best Model Selection")
        st.markdown("""
        Automatically select the best model for your forecasting needs based on historical performance.
        """)
        
        # Model selection criteria
        col1, col2, col3 = st.columns(3)
        
        with col1:
            business_context = st.selectbox(
                "Business Context",
                options=["general", "short_term", "long_term", "promotional", "inventory", "financial"],
                format_func=lambda x: x.replace("_", " ").title(),
                help="Select the business context for your forecast"
            )
        
        with col2:
            horizon = st.number_input(
                "Forecast Horizon (days)",
                min_value=1,
                max_value=365,
                value=30,
                help="Number of days to forecast"
            )
        
        with col3:
            primary_metric = st.selectbox(
                "Primary Metric",
                options=["mae", "rmse", "mape", "r2", "correlation", "coverage"],
                help="Metric to optimize for model selection"
            )
        
        # Advanced filters
        with st.expander("🔧 Advanced Filters", expanded=False):
            col4, col5 = st.columns(2)
            
            with col4:
                model_types = st.multiselect(
                    "Model Types",
                    options=["arima", "sarima", "ets", "prophet", "linear", 
                            "random_forest", "svr", "xgboost", "lstm"],
                    default=[],
                    help="Limit to specific model types"
                )
                if not model_types:
                    model_types = None
            
            with col5:
                weeks = st.multiselect(
                    "Training Weeks",
                    options=[2, 3],
                    default=[],
                    help="Limit to models from specific weeks"
                )
                if not weeks:
                    weeks = None
        
        # Find best models button
        if st.button("🔍 Find Best Models", type="primary"):
            with st.spinner("Evaluating models..."):
                # Run model evaluation
                self.model_selector.evaluate_all_models()
                
                # Get model recommendation
                recommendation = self.model_selector.get_model_recommendation(
                    business_context, horizon
                )
                
                # Get top models
                top_models = self.model_selector.select_best_models(
                    metric=primary_metric,
                    top_k=5,
                    model_types=model_types,
                    weeks=weeks
                )
                
                # Display results
                if recommendation:
                    st.success(f"🎯 **Recommended Model:** {recommendation['label']}")
                    st.info(f"**Reason:** {recommendation.get('reason', 'N/A')}")
                    
                    # Display recommendation metrics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("MAE", f"{recommendation['metrics'].get('mae', 0):.2f}")
                    with col2:
                        st.metric("RMSE", f"{recommendation['metrics'].get('rmse', 0):.2f}")
                    with col3:
                        st.metric("MAPE", f"{recommendation['metrics'].get('mape', 0):.1f}%")
                    with col4:
                        st.metric("R²", f"{recommendation['metrics'].get('r2', 0):.3f}")
                
                # Display top models comparison
                if top_models:
                    st.subheader("🏆 Top Models Comparison")
                    
                    # Create comparison table
                    comparison_data = []
                    for model in top_models:
                        comparison_data.append({
                            "Rank": model["rank"],
                            "Model": model["label"],
                            "Type": model["model_type"],
                            "Week": model["week"],
                            "Score": f"{model['score']:.3f}",
                            "MAE": f"{model['metrics'].get('mae', 0):.2f}",
                            "RMSE": f"{model['metrics'].get('rmse', 0):.2f}",
                            "MAPE": f"{model['metrics'].get('mape', 0):.1f}%"
                        })
                    
                    st.table(pd.DataFrame(comparison_data))
                    
                    # Visualization
                    st.subheader("📈 Model Comparison Visualization")
                    fig = self.model_selector.visualize_comparison()
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
        
        # Quick actions
        st.markdown("---")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📊 View All Model Performance"):
                with st.spinner("Loading performance data..."):
                    df = self.model_selector.create_comparison_report()
                    if not df.empty:
                        st.dataframe(df, use_container_width=True)
                        
                        # Download option
                        csv = df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Performance Report",
                            data=csv,
                            file_name="model_performance_comparison.csv",
                            mime="text/csv"
                        )
        
        with col2:
            if st.button("🔄 Re-evaluate All Models"):
                with st.spinner("Re-evaluating models..."):
                    self.model_selector.evaluate_all_models(force_reload=True)
                    st.success("✅ Model evaluation completed!")
    
    def _run_performance_analytics(self):
        """Run performance analytics interface."""
        st.header("📈 Performance Analytics")
        st.markdown("""
        Track and analyze model performance over time for continuous improvement.
        """)
        
        # Performance metrics overview
        st.subheader("📊 Overall Performance")
        
        # Get top models from tracker
        top_models = self.performance_tracker.get_top_models(top_n=5)
        
        if top_models:
            # Display top models
            col1, col2, col3 = st.columns(3)
            
            with col1:
                best_model = top_models[0]
                st.metric(
                    "Best Model",
                    best_model["model_label"],
                    delta=f"Reliability: {best_model['reliability_score']:.1f}%"
                )
            
            with col2:
                total_runs = sum(m["total_runs"] for m in top_models)
                st.metric("Total Forecast Runs", total_runs)
            
            with col3:
                avg_success = np.mean([m["success_rate"] for m in top_models])
                st.metric("Average Success Rate", f"{avg_success:.1f}%")
            
            # Performance trends
            st.subheader("📈 Performance Trends")
            
            # Select model for detailed analysis
            model_options = {m["model_key"]: m["model_label"] for m in top_models}
            selected_model = st.selectbox(
                "Select Model for Detailed Analysis",
                options=list(model_options.keys()),
                format_func=lambda x: model_options[x]
            )
            
            if selected_model:
                # Get model performance
                performance = self.performance_tracker.get_model_performance(selected_model)
                
                if "error" not in performance:
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Total Runs", performance["total_runs"])
                    
                    with col2:
                        st.metric("Success Rate", f"{performance['success_rate']:.1f}%")
                    
                    with col3:
                        st.metric("Avg MAE", f"{performance.get('avg_mae', 0):.2f}")
                    
                    with col4:
                        st.metric("Best MAE", f"{performance.get('best_mae', 0):.2f}")
                    
                    # Get performance trends
                    trends = self.performance_tracker.get_performance_trends(
                        selected_model, metric="mae"
                    )
                    
                    if not trends.empty:
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=trends["timestamp"],
                            y=trends["mae"],
                            mode="lines+markers",
                            name="MAE",
                            line=dict(color="red", width=2)
                        ))
                        
                        fig.add_trace(go.Scatter(
                            x=trends["timestamp"],
                            y=trends["mae_rolling"],
                            mode="lines",
                            name="7-Day Average",
                            line=dict(color="blue", width=2, dash="dash")
                        ))
                        
                        fig.update_layout(
                            title=f"MAE Trend for {model_options[selected_model]}",
                            xaxis_title="Date",
                            yaxis_title="MAE",
                            height=400
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
        
        # Generate report
        st.markdown("---")
        if st.button("📋 Generate Performance Report", type="primary"):
            with st.spinner("Generating report..."):
                report_path = self.performance_tracker.generate_performance_report()
                st.success(f"✅ Report generated: {report_path}")
                
                # Provide download link
                with open(report_path, 'r') as f:
                    html_content = f.read()
                
                st.download_button(
                    label="📥 Download HTML Report",
                    data=html_content,
                    file_name="performance_report.html",
                    mime="text/html"
                )
    
    def _run_batch_forecasting(self):
        """Run batch forecasting interface."""
        st.header("🔢 Batch Forecasting")
        st.markdown("""
        Generate forecasts for multiple items/stores in batch mode.
        """)
        from app.components.batch_forecast import BatchForecaster
        # Configuration
        st.subheader("⚙️ Batch Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Item selection
            item_input = st.text_area(
                "Item IDs (comma-separated)",
                value="105577, 105578, 105579",
                help="Enter item IDs separated by commas"
            )
            
            # Parse items
            item_ids = [item.strip() for item in item_input.split(",") if item.strip()]
        
        with col2:
            # Store selection
            store_input = st.text_area(
                "Store IDs (comma-separated)",
                value="24, 25, 26",
                help="Enter store IDs separated by commas"
            )
            
            # Parse stores
            store_ids = [store.strip() for store in store_input.split(",") if store.strip()]
        
        # Generate all combinations
        item_store_pairs = []
        for item_id in item_ids:
            for store_id in store_ids:
                item_store_pairs.append((item_id, store_id))
        
        st.info(f"📊 Will forecast {len(item_store_pairs)} item-store combinations")
        
        # Model selection
        st.subheader("🤖 Model Selection")
        
        model_key = st.selectbox(
            "Select Model for Batch",
            options=list(MODEL_REGISTRY.keys()),
            format_func=lambda k: MODEL_REGISTRY[k]["label"],
            help="Model to use for all forecasts in batch"
        )
        
        # Forecast configuration
        st.subheader("🔧 Forecast Settings")
        
        col3, col4 = st.columns(2)
        
        with col3:
            horizon = st.number_input(
                "Forecast Horizon",
                min_value=1,
                max_value=365,
                value=30,
                key="batch_horizon"
            )
            
            start_date = st.date_input(
                "Start Date",
                value=pd.Timestamp.today(),
                key="batch_start_date"
            )
        
        with col4:
            frequency = st.selectbox(
                "Frequency",
                options=["D", "W", "M"],
                format_func=lambda x: {"D": "Daily", "W": "Weekly", "M": "Monthly"}[x],
                key="batch_frequency"
            )
            
            onpromotion = st.selectbox(
                "On Promotion?",
                options=[0, 1],
                format_func=lambda x: "Yes" if x == 1 else "No",
                key="batch_promotion"
            )
        
        # Execution options
        st.subheader("⚡ Execution Options")
        
        col5, col6 = st.columns(2)
        
        with col5:
            parallel = st.checkbox(
                "Parallel Execution",
                value=True,
                help="Run forecasts in parallel (faster)"
            )
        
        with col6:
            max_workers = st.slider(
                "Max Workers",
                min_value=1,
                max_value=10,
                value=4,
                disabled=not parallel,
                help="Maximum parallel workers"
            )
        
        # Progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Run batch forecast button
        if st.button("🚀 Run Batch Forecast", type="primary", use_container_width=True):
            if not item_store_pairs:
                st.error("Please specify at least one item and store")
                return
            
            # Prepare config
            config = {
                "horizon": horizon,
                "freq": frequency,
                "start_date": start_date,
                "onpromotion": onpromotion,
                "show_confidence": True,
                "confidence_level": 0.95
            }
            
            # Initialize batch forecaster
           
            
            self.batch_forecaster = BatchForecaster(
                forecast_engine_class=ForecastEngine,
                historical_data_path=get_path("filtered")
            )
            
            # Set session state
            st.session_state.batch_in_progress = True
            st.session_state.batch_progress = 0.0
            
            def update_progress(completed, total):
                progress = completed / total
                st.session_state.batch_progress = progress
                progress_bar.progress(progress)
                status_text.text(f"Processed {completed}/{total} items")
            
            # Run batch forecast
            try:
                results = self.batch_forecaster.forecast_batch(
                    item_store_pairs=item_store_pairs,
                    config=config,
                    model_key=model_key,
                    parallel=parallel,
                    max_workers=max_workers,
                    progress_callback=update_progress
                )
                
                # Reset progress
                st.session_state.batch_in_progress = False
                progress_bar.empty()
                
                # Display results
                successful = sum(1 for r in results.values() if r.get("status") == "success")
                failed = len(results) - successful
                
                st.success(f"✅ Batch forecast completed!")  # noqa: F541
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Successful", successful)
                with col2:
                    st.metric("Failed", failed)
                
                # Show detailed results
                with st.expander("📋 View Detailed Results", expanded=False):
                    results_df = pd.DataFrame([
                        {
                            "Item": r["item_id"],
                            "Store": r["store_id"],
                            "Status": r["status"],
                            "Total Forecast": r.get("summary_stats", {}).get("total_forecast", 0),
                            "Error": r.get("error", "N/A")
                        }
                        for r in results.values()
                    ])
                    
                    st.dataframe(results_df, use_container_width=True)
                
                # Generate report
                if successful > 0:
                    if st.button("📊 Generate Batch Report"):
                        with st.spinner("Generating report..."):
                            batches = self.batch_forecaster.get_available_batches()
                            if batches:
                                latest_batch = batches[0]["batch_id"]
                                report_path = self.batch_forecaster.generate_batch_report(latest_batch)
                                
                                with open(report_path, 'r') as f:
                                    html_content = f.read()
                                
                                st.download_button(
                                    label="📥 Download Batch Report",
                                    data=html_content,
                                    file_name=f"batch_report_{latest_batch}.html",
                                    mime="text/html"
                                )
                
            except Exception as e:
                st.error(f"❌ Batch forecast failed: {str(e)}")
                st.session_state.batch_in_progress = False
        
        # Display available batches
        st.markdown("---")
        st.subheader("📁 Available Batch Forecasts")
        
        if self.batch_forecaster:
            batches = self.batch_forecaster.get_available_batches()
            
            if batches:
                for batch in batches[:5]:  # Show only recent 5
                    with st.expander(f"Batch: {batch['batch_id']}"):
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Items", batch["item_count"])
                        with col2:
                            st.metric("Success Rate", f"{batch['success_rate']:.1f}%")
                        with col3:
                            st.metric("Model", batch["model_key"])
                        
                        if st.button(f"Load Batch {batch['batch_id']}", key=batch['batch_id']):
                            batch_data = self.batch_forecaster.load_batch_results(batch['batch_id'])
                            if batch_data:
                                st.json(batch_data.get("summary", {}))
            else:
                st.info("No batch forecasts available. Run a batch forecast to see results here.")


# ============================================================================
# ENTRY POINT (for standalone testing)
# ============================================================================
if __name__ == "__main__":
    app = Forecast()
    app.run()
