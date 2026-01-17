# app/utils/visualizer.py
"""
Visualization module for forecast results
"""
from app.bootstrap import *  # ensures project root is on sys.path  # noqa: F403 
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px  # noqa: F401
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import matplotlib.dates as mdates  # noqa: F401
from typing import Dict, List, Optional, Tuple, Any  # noqa: F401



from pathlib import Path



class ForecastVisualizer:
    """
    Visualization tools for forecast results.
    """

    # Color scheme for consistent visualizations
    COLORS = {
        'primary': '#1f77b4',
        'accent': '#ff7f0e',
        'secondary': '#2ca02c',
        'tertiary': '#d62728',
        'quaternary': '#9467bd',
        'background': '#f0f0f0'
    }

    def __init__(self, historical_df: Optional[pd.DataFrame] = None):
        """
        Initialize visualizer.

        Args:
            historical_df: Optional historical data for context
        """
        self.historical_df = historical_df

    def _save_fig(self, filename: str):
        """
        Save matplotlib figure to file.

        Args:
            filename: Output filename
        """
        Path("visualizations").mkdir(parents=True, exist_ok=True)
        plt.savefig(f"visualizations/{filename}", dpi=300, bbox_inches='tight')

    def save_figure(self, fig: Any, filename: str, format: str = "png"):
        """
        Save a Plotly or Matplotlib figure to disk.

        Args:
            fig: Figure object (Plotly or Matplotlib)
            filename: Base filename without extension
            format: File format (e.g., 'png')
        """
        Path("visualizations").mkdir(parents=True, exist_ok=True)
        path = Path("visualizations") / f"{filename}.{format}"

        # Plotly figure
        if isinstance(fig, go.Figure):
            try:
                fig.write_image(str(path))
            except Exception:
                # Fallback: export as HTML if static export is not available
                html_path = Path("visualizations") / f"{filename}.html"
                fig.write_html(str(html_path))
        else:
            # Assume Matplotlib
            fig.savefig(path, dpi=300, bbox_inches="tight")

    def plot_forecast_timeseries(
        self,
        forecast_df: pd.DataFrame,
        historical_df: Optional[pd.DataFrame] = None,
        title: str = "Forecast vs Historical",
        show_confidence: bool = True,
        show_points: bool = False,
        height: int = 500,
        width: Optional[int] = None,
        confidence_level: float = 0.95
    ) -> go.Figure:
        """
        Plot forecast time series with historical context.

        Args:
            forecast_df: DataFrame with forecast results
            historical_df: Historical data (uses self.historical_df if None)
            title: Plot title
            show_confidence: Whether to show confidence intervals
            show_points: Whether to show individual points
            height: Plot height
            width: Plot width (None = auto)
            confidence_level: Confidence level for intervals (e.g., 0.95)

        Returns:
            Plotly figure
        """
        if historical_df is None:
            historical_df = self.historical_df

        fig = go.Figure()

        # Add historical data
        if historical_df is not None and "unit_sales" in historical_df.columns and "date" in historical_df.columns:
            hist_tail = historical_df.tail(min(365, len(historical_df)))
            fig.add_trace(go.Scatter(
                x=hist_tail["date"],
                y=hist_tail["unit_sales"],
                mode="lines",
                name="Historical",
                line=dict(color=self.COLORS['primary'], width=2),
                opacity=0.8
            ))

        # Determine forecast column names
        forecast_col = "point_forecast" if "point_forecast" in forecast_df.columns else "forecast"
        date_col = "date" if "date" in forecast_df.columns else forecast_df.index.name

        if date_col not in forecast_df.columns:
            date_col = next((col for col in forecast_df.columns if "date" in col.lower()), None)
            if date_col is None:
                raise ValueError("No date column found in forecast DataFrame")

        # Add forecast
        mode = "lines+markers" if show_points else "lines"
        fig.add_trace(go.Scatter(
            x=forecast_df[date_col],
            y=forecast_df[forecast_col],
            mode=mode,
            name="Forecast",
            line=dict(color=self.COLORS['accent'], width=3, dash="dash"),
            marker=dict(size=6) if show_points else None
        ))

        # Add confidence intervals if available
        if show_confidence and "lower_bound" in forecast_df.columns and "upper_bound" in forecast_df.columns:
            fig.add_trace(go.Scatter(
                x=forecast_df[date_col].tolist() + forecast_df[date_col].tolist()[::-1],
                y=forecast_df["upper_bound"].tolist() + forecast_df["lower_bound"].tolist()[::-1],
                fill='toself',
                fillcolor=f'rgba{tuple(int(self.COLORS["accent"].lstrip("#")[i:i+2], 16) for i in (0, 2, 4)) + (0.2,)}',
                line=dict(color='rgba(255,255,255,0)'),
                hoverinfo="skip",
                showlegend=True,
                name=f"{confidence_level*100:.0f}% Confidence"
            ))

        # Add forecast start line (fix Timestamp arithmetic issue)
        if len(forecast_df[date_col]) > 0:
            forecast_start = pd.to_datetime(forecast_df[date_col].iloc[0]).to_pydatetime()
            fig.add_vline(
            x=forecast_start,
            line_dash="dash",
            line_color="red",
            opacity=0.7
        )

        fig.add_annotation(
            x=forecast_start,
            y=1,
            yref="paper",
            text="Forecast Start",
            showarrow=True,
            arrowhead=2,
            ax=20,
            ay=-30
        )


        # Update layout
        fig.update_layout(
            title=dict(
                text=f"{title}<br><sup>{confidence_level*100:.0f}% Confidence Interval (if available)</sup>",
                x=0.5,
                font=dict(size=16)
            ),
            xaxis_title="Date",
            yaxis_title="Unit Sales",
            hovermode="x unified",
            height=height,
            width=width,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
                bgcolor="rgba(255, 255, 255, 0.8)"
            ),
            plot_bgcolor="white",
            xaxis=dict(
                gridcolor="lightgray",
                showgrid=True
            ),
            yaxis=dict(
                gridcolor="lightgray",
                showgrid=True
            )
        )

        return fig

    def plot_forecast_distribution_plotly(
        self,
        forecast_df: pd.DataFrame,
        forecast_col: str = "point_forecast",
        title: str = "Forecast Distribution",
        height: int = 400
    ) -> go.Figure:
        """
        Plot distribution of forecast values.

        Args:
            forecast_df: DataFrame with forecast values
            forecast_col: Name of forecast column
            title: Plot title
            height: Plot height

        Returns:
            Plotly figure
        """
        if forecast_col not in forecast_df.columns:
            raise ValueError(f"Column '{forecast_col}' not found in forecast DataFrame")

        fig = go.Figure()

        fig.add_trace(go.Histogram(
            x=forecast_df[forecast_col],
            nbinsx=30,
            name="Forecast Distribution",
            marker_color=self.COLORS['secondary'],
            opacity=0.7
        ))

        # Add vertical line for mean
        mean_val = forecast_df[forecast_col].mean()
        fig.add_vline(
            x=mean_val,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Mean: {mean_val:.2f}",
            annotation_position="top right"
        )

        fig.update_layout(
            title=dict(text=title, x=0.5),
            xaxis_title="Forecast Value",
            yaxis_title="Frequency",
            height=height,
            plot_bgcolor="white",
            bargap=0.05
        )

        return fig

    def plot_forecast_uncertainty(
        self,
        forecast_df: pd.DataFrame,
        title: str = "Forecast Uncertainty",
        height: int = 500,
        confidence_level: float = 0.95
    ) -> go.Figure:
        """
        Plot forecast with uncertainty bands.

        Args:
            forecast_df: DataFrame with forecast and bounds
            title: Plot title
            height: Plot height
            confidence_level: Confidence level for intervals

        Returns:
            Plotly figure
        """
        if not all(col in forecast_df.columns for col in ["lower_bound", "upper_bound"]):
            raise ValueError("DataFrame must contain 'lower_bound' and 'upper_bound' columns")

        fig = go.Figure()

        date_col = "date" if "date" in forecast_df.columns else forecast_df.index.name

        # Add uncertainty bands
        fig.add_trace(go.Scatter(
            x=forecast_df[date_col],
            y=forecast_df["upper_bound"],
            mode="lines",
            line=dict(width=0),
            showlegend=False,
            name="Upper Bound"
        ))

        fig.add_trace(go.Scatter(
            x=forecast_df[date_col],
            y=forecast_df["lower_bound"],
            mode="lines",
            line=dict(width=0),
            fillcolor=f'rgba{tuple(int(self.COLORS["accent"].lstrip("#")[i:i+2], 16) for i in (0, 2, 4)) + (0.2,)}',
            fill="tonexty",
            showlegend=True,
            name=f"{confidence_level*100:.0f}% Uncertainty Band"
        ))

        # Add point forecast
        forecast_col = "point_forecast" if "point_forecast" in forecast_df.columns else "forecast"
        fig.add_trace(go.Scatter(
            x=forecast_df[date_col],
            y=forecast_df[forecast_col],
            mode="lines",
            name="Point Forecast",
            line=dict(color=self.COLORS['accent'], width=3)
        ))

        # Calculate and display uncertainty metrics
        avg_uncertainty = (forecast_df["upper_bound"] - forecast_df["lower_bound"]).mean()
        uncertainty_ratio = avg_uncertainty / forecast_df[forecast_col].mean() if forecast_df[forecast_col].mean() != 0 else 0

        fig.update_layout(
            title=dict(
                text=f"{title}<br><sup>Avg Uncertainty: ±{avg_uncertainty/2:.2f} ({uncertainty_ratio*100:.1f}%) at {confidence_level*100:.0f}% CI</sup>",
                x=0.5
            ),
            xaxis_title="Date",
            yaxis_title="Unit Sales",
            height=height,
            hovermode="x unified",
            plot_bgcolor="white",
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            )
        )

        return fig

    def plot_forecast_components(
        self,
        forecast_df: pd.DataFrame,
        component_cols: List[str],
        title: str = "Forecast Components",
        height: int = 400,
        cols_per_row: int = 2
    ) -> go.Figure:
        """
        Plot forecast components (trend, seasonality, etc.).

        Args:
            forecast_df: DataFrame with forecast and components
            component_cols: List of component column names
            title: Plot title
            height: Height per subplot
            cols_per_row: Number of columns in subplot grid

        Returns:
            Plotly figure with subplots
        """
        n_components = len(component_cols)
        if n_components == 0:
            raise ValueError("No component columns provided")

        n_rows = (n_components + cols_per_row - 1) // cols_per_row

        fig = make_subplots(
            rows=n_rows,
            cols=cols_per_row,
            subplot_titles=component_cols,
            vertical_spacing=0.1,
            horizontal_spacing=0.1
        )

        date_col = "date" if "date" in forecast_df.columns else forecast_df.index.name

        for i, component in enumerate(component_cols):
            if component in forecast_df.columns:
                row = i // cols_per_row + 1
                col = i % cols_per_row + 1

                fig.add_trace(
                    go.Scatter(
                        x=forecast_df[date_col] if date_col in forecast_df.columns else forecast_df.index,
                        y=forecast_df[component],
                        mode="lines",
                        name=component,
                        line=dict(color=self.COLORS['primary'], width=2)
                    ),
                    row=row,
                    col=col
                )

        fig.update_layout(
            title=dict(text=title, x=0.5),
            height=height * n_rows,
            showlegend=False,
            plot_bgcolor="white"
        )

        return fig

    def plot_forecast_heatmap(
        self,
        forecast_df: pd.DataFrame,
        title: str = "Monthly Forecast Heatmap",
        height: int = 500
    ) -> go.Figure:
        """
        Plot a heatmap of forecast values by month.

        Args:
            forecast_df: DataFrame with forecast results
            title: Plot title
            height: Plot height

        Returns:
            Plotly figure
        """
        if "month" not in forecast_df.columns and "date" in forecast_df.columns:
            forecast_df["month"] = pd.to_datetime(forecast_df["date"]).dt.month

        if "month" not in forecast_df.columns:
            raise ValueError("No month column found in forecast DataFrame")

        # Pivot the data for heatmap
        heatmap_data = forecast_df.pivot_table(
            index="month",
            columns=forecast_df["date"].dt.day,
            values="point_forecast",
            aggfunc="mean"
        )

        fig = go.Figure(data=go.Heatmap(
            z=heatmap_data.values,
            x=heatmap_data.columns,
            y=heatmap_data.index,
            colorscale='Viridis',
            text=heatmap_data.values,
            textfont={"size": 10}
        ))

        fig.update_layout(
            title=dict(text=title, x=0.5),
            xaxis_title="Day of Month",
            yaxis_title="Month",
            height=height,
            plot_bgcolor="white",
            xaxis=dict(tickmode='linear'),
            yaxis=dict(tickmode='linear')
        )

        return fig

    def plot_forecast_waterfall(
        self,
        forecast_df: pd.DataFrame,
        start_value: float,
        forecast_col: str = "point_forecast",
        title: str = "Forecast Waterfall",
        height: int = 500
    ) -> go.Figure:
        """
        Create waterfall chart showing forecast changes.

        Args:
            forecast_df: DataFrame with forecast values
            start_value: Starting value for waterfall
            forecast_col: Name of forecast column
            title: Plot title
            height: Plot height

        Returns:
            Plotly waterfall figure
        """
        if forecast_col not in forecast_df.columns:
            raise ValueError(f"Column '{forecast_col}' not found in forecast DataFrame")

        # Calculate changes
        changes = forecast_df[forecast_col].diff().fillna(forecast_df[forecast_col].iloc[0] - start_value)

        fig = go.Figure(go.Waterfall(
            name="Forecast",
            orientation="v",
            measure=["absolute"] + ["relative"] * (len(changes) - 1),
            x=forecast_df["date"] if "date" in forecast_df.columns else forecast_df.index,
            textposition="outside",
            text=[f"{c:+.2f}" for c in changes],
            y=changes.tolist(),
            connector={"line": {"color": "rgb(63, 63, 63)"}},
            increasing={"marker": {"color": "green"}},
            decreasing={"marker": {"color": "red"}}
        ))

        fig.update_layout(
            title=dict(text=title, x=0.5),
            xaxis_title="Date",
            yaxis_title="Change",
            height=height,
            plot_bgcolor="white",
            showlegend=False
        )

        return fig

    def plot_forecast(
        self,
        forecast_results: Dict,
        historical_points: int = 100,
        figsize: Tuple[int, int] = (15, 8)
    ) -> plt.Figure:
        """
        Visualize future forecasts with historical context.

        Args:
            forecast_results: Forecast results from forecast_future()
            historical_points: Number of historical points to show
            figsize: Figure size (width, height)

        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 1, figsize=figsize)

        # 1. Historical vs Forecast Plot
        ax1 = axes[0]

        # Plot historical data if available
        if self.historical_df is not None and "date" in self.historical_df.columns:
            historical_data = self.historical_df.sort_values("date").tail(historical_points)
            if "unit_sales" in historical_data.columns:
                ax1.plot(
                    historical_data["date"],
                    historical_data["unit_sales"],
                    color=self.COLORS['primary'],
                    linewidth=2,
                    label='Historical'
                )

        # Plot forecasts
        forecast_dates = pd.to_datetime(forecast_results.get("forecast_dates", forecast_results.get("date")))
        point_forecasts = forecast_results.get("point_forecasts", forecast_results.get("forecast"))
        lower_bounds = forecast_results.get("lower_bounds")
        upper_bounds = forecast_results.get("upper_bounds")

        ax1.plot(forecast_dates, point_forecasts,
                 color=self.COLORS['accent'],
                 linewidth=3,
                 label='Forecast')

        # Add confidence interval if available
        if lower_bounds is not None and upper_bounds is not None:
            ax1.fill_between(forecast_dates, lower_bounds, upper_bounds,
                             color=self.COLORS['accent'], alpha=0.2,
                             label=f"{forecast_results.get('confidence_level', 0.95)*100:.0f}% Confidence")

        # Add forecast start line
        ax1.axvline(x=forecast_dates[0], color='red', linestyle='--', alpha=0.7)
        ax1.text(forecast_dates[0], ax1.get_ylim()[1] * 0.95,
                 'Forecast Start', rotation=90, verticalalignment='top')

        ax1.set_title(f"Random Forest Forecast - {forecast_results.get('model_variant', 'Model')}",
                      fontsize=14, fontweight='bold')
        ax1.set_xlabel("Date", fontsize=12)
        ax1.set_ylabel("Unit Sales", fontsize=12)
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')

        # 2. Uncertainty Plot (if confidence intervals available)
        ax2 = axes[1]
        if lower_bounds is not None and upper_bounds is not None:
            uncertainty = upper_bounds - lower_bounds

            ax2.bar(forecast_dates, uncertainty,
                    color=self.COLORS['secondary'],
                    alpha=0.6, width=0.8)

            ax2.set_title("Forecast Uncertainty Over Time", fontsize=14, fontweight='bold')
            ax2.set_xlabel("Date", fontsize=12)
            ax2.set_ylabel("Uncertainty Range", fontsize=12)
        else:
            # Alternative: Plot forecast values distribution over time
            ax2.plot(forecast_dates, point_forecasts,
                     color=self.COLORS['tertiary'],
                     linewidth=2,
                     marker='o',
                     markersize=4)
            ax2.fill_between(forecast_dates, point_forecasts * 0.9, point_forecasts * 1.1,
                             color=self.COLORS['tertiary'], alpha=0.2)
            ax2.set_title("Forecast Values with 10% Variation", fontsize=14, fontweight='bold')
            ax2.set_xlabel("Date", fontsize=12)
            ax2.set_ylabel("Unit Sales", fontsize=12)

        ax2.grid(True, alpha=0.3, axis='y')
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')

        plt.tight_layout()
        return fig

    def plot_forecast_distribution(self, forecast_results: Dict, figsize: Tuple[int, int] = (12, 8)):
        """
        Plot distribution of forecasts.

        Args:
            forecast_results: Forecast results from forecast_future()
            figsize: Figure size

        Returns:
            Matplotlib figure
        """
        point_forecasts = forecast_results.get("point_forecasts", forecast_results.get("forecast"))
        lower_bounds = forecast_results.get("lower_bounds")
        upper_bounds = forecast_results.get("upper_bounds")

        fig, axes = plt.subplots(1, 2, figsize=figsize)

        # 1. Forecast distribution histogram
        ax1 = axes[0]
        ax1.hist(point_forecasts, bins=20,
                 color=self.COLORS['secondary'],
                 alpha=0.7, edgecolor='black', density=True)

        # Add normal distribution fit
        mu, sigma = np.mean(point_forecasts), np.std(point_forecasts)
        x = np.linspace(point_forecasts.min(), point_forecasts.max(), 100)
        ax1.plot(x, 1/(sigma * np.sqrt(2 * np.pi)) *
                 np.exp(-0.5 * ((x - mu) / sigma) ** 2),
                 'r-', linewidth=2, label=f'Normal(μ={mu:.2f}, σ={sigma:.2f})')

        ax1.axvline(mu, color='green', linestyle='--', label=f'Mean: {mu:.2f}')
        ax1.axvline(np.median(point_forecasts), color='orange',
                    linestyle='--', label=f'Median: {np.median(point_forecasts):.2f}')

        ax1.set_title("Forecast Distribution", fontsize=12, fontweight='bold')
        ax1.set_xlabel("Unit Sales", fontsize=11)
        ax1.set_ylabel("Density", fontsize=11)
        ax1.legend(fontsize=9)
        ax1.grid(True, alpha=0.3)

        # 2. Uncertainty over forecast horizon or Forecast timeline
        ax2 = axes[1]
        if lower_bounds is not None and upper_bounds is not None:
            uncertainty = upper_bounds - lower_bounds

            ax2.plot(range(1, len(uncertainty) + 1), uncertainty,
                     color=self.COLORS['accent'], linewidth=2)
            ax2.fill_between(range(1, len(uncertainty) + 1), 0, uncertainty,
                             color=self.COLORS['accent'], alpha=0.2)

            ax2.set_title("Uncertainty Over Forecast Horizon",
                          fontsize=12, fontweight='bold')
            ax2.set_xlabel("Forecast Period", fontsize=11)
            ax2.set_ylabel("Uncertainty Range", fontsize=11)
        else:
            # Alternative: Plot forecast progression
            forecast_dates = pd.to_datetime(forecast_results.get("forecast_dates", forecast_results.get("date")))
            ax2.plot(forecast_dates, point_forecasts,
                     color=self.COLORS['quaternary'],
                     linewidth=2, marker='o', markersize=4)
            ax2.set_title("Forecast Progression",
                          fontsize=12, fontweight='bold')
            ax2.set_xlabel("Date", fontsize=11)
            ax2.set_ylabel("Unit Sales", fontsize=11)
            plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')

        ax2.grid(True, alpha=0.3)

        # Add statistics text
        stats_text = f"""
        Forecast Statistics:
        Horizon: {len(point_forecasts)} periods
        Mean: {mu:.2f}
        Std: {sigma:.2f}
        Min: {point_forecasts.min():.2f}
        Max: {point_forecasts.max():.2f}
        Total: {point_forecasts.sum():.2f}
        """

        if lower_bounds is not None and upper_bounds is not None:
            uncertainty = upper_bounds - lower_bounds
            stats_text += f"Mean Uncertainty: {uncertainty.mean():.2f}"

        plt.figtext(0.02, 0.02, stats_text, fontsize=9,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.suptitle(f"Random Forest Forecast Analysis - {forecast_results.get('model_variant', 'Model')}",
                     fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()

        return fig

    def plot_forecast_vs_actual(
        self,
        forecast_df: pd.DataFrame,
        actual_df: pd.DataFrame,
        forecast_col: str = "forecast",
        actual_col: str = "unit_sales",
        title: str = "Forecast vs Actual",
        height: int = 500
    ) -> go.Figure:
        """
        Plot forecast vs actual values.

        Args:
            forecast_df: DataFrame with forecast values
            actual_df: DataFrame with actual values
            forecast_col: Name of forecast column
            actual_col: Name of actual column
            title: Plot title
            height: Plot height

        Returns:
            Plotly figure
        """
        # Merge forecast and actual on date
        date_col = "date" if "date" in forecast_df.columns else "index"

        if date_col not in forecast_df.columns:
            forecast_df = forecast_df.reset_index()

        if date_col not in actual_df.columns:
            actual_df = actual_df.reset_index()

        merged = pd.merge(
            forecast_df[[date_col, forecast_col]],
            actual_df[[date_col, actual_col]],
            on=date_col,
            how="inner",
            suffixes=("_forecast", "_actual")
        )

        fig = go.Figure()

        # Add actual values
        fig.add_trace(go.Scatter(
            x=merged[date_col],
            y=merged[actual_col],
            mode="lines+markers",
            name="Actual",
            line=dict(color=self.COLORS['primary'], width=2),
            marker=dict(size=6)
        ))

        # Add forecast values
        fig.add_trace(go.Scatter(
            x=merged[date_col],
            y=merged[forecast_col],
            mode="lines+markers",
            name="Forecast",
            line=dict(color=self.COLORS['accent'], width=2, dash="dash"),
            marker=dict(size=6, symbol="x")
        ))

        # Calculate and display error metrics
        errors = merged[actual_col] - merged[forecast_col]
        mae = errors.abs().mean()
        rmse = np.sqrt((errors**2).mean())
        mape = (errors.abs() / merged[actual_col]).mean() * 100

        fig.update_layout(
            title=dict(
                text=f"{title}<br><sup>MAE: {mae:.2f} | RMSE: {rmse:.2f} | MAPE: {mape:.1f}%</sup>",
                x=0.5
            ),
            xaxis_title="Date",
            yaxis_title="Unit Sales",
            height=height,
            hovermode="x unified",
            plot_bgcolor="white",
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            )
        )

        return fig

    def plot_forecast_error_analysis(
        self,
        forecast_df: pd.DataFrame,
        actual_df: pd.DataFrame,
        forecast_col: str = "forecast",
        actual_col: str = "unit_sales",
        title: str = "Forecast Error Analysis",
        height: int = 600
    ) -> go.Figure:
        """
        Comprehensive forecast error analysis.

        Args:
            forecast_df: DataFrame with forecast values
            actual_df: DataFrame with actual values
            forecast_col: Name of forecast column
            actual_col: Name of actual column
            title: Plot title
            height: Plot height

        Returns:
            Plotly figure with subplots
        """
        # Merge data
        date_col = "date" if "date" in forecast_df.columns else "index"

        if date_col not in forecast_df.columns:
            forecast_df = forecast_df.reset_index()

        if date_col not in actual_df.columns:
            actual_df = actual_df.reset_index()

        merged = pd.merge(
            forecast_df[[date_col, forecast_col]],
            actual_df[[date_col, actual_col]],
            on=date_col,
            how="inner"
        )

        # Calculate errors
        merged["error"] = merged[actual_col] - merged[forecast_col]
        merged["abs_error"] = merged["error"].abs()
        merged["pct_error"] = (merged["error"] / merged[actual_col]) * 100

        # Create subplots
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=[
                "Error Distribution",
                "Error vs Forecast",
                "Cumulative Error",
                "Error Over Time"
            ],
            vertical_spacing=0.15,
            horizontal_spacing=0.1
        )

        # 1. Error distribution
        fig.add_trace(
            go.Histogram(
                x=merged["error"],
                nbinsx=30,
                name="Error Distribution",
                marker_color=self.COLORS['tertiary'],
                opacity=0.7
            ),
            row=1,
            col=1
        )

        # Add mean error line
        mean_error = merged["error"].mean()
        fig.add_vline(
            x=mean_error,
            line_dash="dash",
            line_color="red",
            row=1,
            col=1
        )

        # 2. Error vs forecast
        fig.add_trace(
            go.Scatter(
                x=merged[forecast_col],
                y=merged["error"],
                mode="markers",
                name="Error vs Forecast",
                marker=dict(
                    color=merged["abs_error"],
                    colorscale="RdYlBu_r",
                    size=8,
                    showscale=True,
                    colorbar=dict(title="Abs Error")
                )
            ),
            row=1,
            col=2
        )

        # Add zero line
        fig.add_hline(y=0, line_dash="dot", line_color="gray", row=1, col=2)

        # 3. Cumulative error
        merged["cumulative_error"] = merged["error"].cumsum()

        fig.add_trace(
            go.Scatter(
                x=merged[date_col],
                y=merged["cumulative_error"],
                mode="lines",
                name="Cumulative Error",
                line=dict(color=self.COLORS['secondary'], width=2)
            ),
            row=2,
            col=1
        )

        # 4. Error over time
        fig.add_trace(
            go.Scatter(
                x=merged[date_col],
                y=merged["error"],
                mode="lines+markers",
                name="Error Over Time",
                line=dict(color=self.COLORS['quaternary'], width=2),
                marker=dict(size=6)
            ),
            row=2,
            col=2
        )

        fig.add_hline(y=0, line_dash="dot", line_color="gray", row=2, col=2)

        fig.update_layout(
            title=dict(text=title, x=0.5),
            height=height,
            showlegend=False,
            plot_bgcolor="white"
        )

        return fig


    def _plot_forecast_historical_comparison(self, forecast_results: Dict):
        """
        Plot forecast comparison with historical patterns.
        
        Args:
            forecast_results: Forecast results dictionary
        """
        if self.historical_df is None or "date" not in self.historical_df.columns:
            return
        
        # Get historical data for comparison
        historical_data = self.historical_df.sort_values("date")
        
        # Extract seasonal patterns if enough data
        if len(historical_data) > 365:  # At least a year of data
            historical_data['month'] = historical_data['date'].dt.month
            
            # Calculate monthly averages
            monthly_avg = historical_data.groupby('month')["unit_sales"].mean()
            
            # Compare forecast with historical pattern
            forecast_dates = pd.to_datetime(forecast_results.get("forecast_dates", forecast_results.get("date")))
            forecast_months = forecast_dates.month
            
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Plot historical monthly pattern
            ax.plot(monthly_avg.index, monthly_avg.values,
                   color=self.COLORS['primary'],
                   linewidth=2, marker='o', label='Historical Monthly Average')
            
            # Plot forecast monthly values
            point_forecasts = forecast_results.get("point_forecasts", forecast_results.get("forecast"))
            forecast_by_month = []
            for month in range(1, 13):
                mask = forecast_months == month
                if any(mask):
                    forecast_by_month.append(np.mean(np.array(point_forecasts)[mask]))
                else:
                    forecast_by_month.append(np.nan)
            
            ax.plot(range(1, 13), forecast_by_month,
                   color=self.COLORS['accent'],
                   linewidth=2, marker='s', label='Forecast Monthly Average')
            
            ax.set_title("Forecast vs Historical Seasonal Pattern", 
                        fontsize=14, fontweight='bold')
            ax.set_xlabel("Month", fontsize=12)
            ax.set_ylabel("Unit Sales", fontsize=12)
            ax.set_xticks(range(1, 13))
            ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            return fig
        return None
    
    def _print_forecast_summary_table(self, forecast_results: Dict):
        """
        Print formatted forecast summary table.
        
        Args:
            forecast_results: Forecast results dictionary
        """
        stats = forecast_results.get("statistics", {})
        
        print("\n" + "=" * 80)
        print("FORECAST SUMMARY TABLE")
        print("=" * 80)
        
        summary_data = [
            ["Model Variant", forecast_results.get("model_variant", "Model")],
            ["Forecast Horizon", f"{forecast_results.get('forecast_horizon', len(forecast_results.get('forecast', [])))} periods"],
            ["Frequency", forecast_results.get("frequency", "Daily")],
            ["Confidence Level", f"{forecast_results.get('confidence_level', 0.95)*100:.1f}%"],
            ["Mean Forecast", f"{stats.get('mean_forecast', np.mean(forecast_results.get('forecast', [0]))):.2f}"],
            ["Median Forecast", f"{stats.get('median_forecast', np.median(forecast_results.get('forecast', [0]))):.2f}"],
            ["Std Forecast", f"{stats.get('std_forecast', np.std(forecast_results.get('forecast', [0]))):.2f}"],
            ["Min Forecast", f"{stats.get('min_forecast', np.min(forecast_results.get('forecast', [0]))):.2f}"],
            ["Max Forecast", f"{stats.get('max_forecast', np.max(forecast_results.get('forecast', [0]))):.2f}"],
            ["Total Forecast", f"{stats.get('total_forecast', np.sum(forecast_results.get('forecast', [0]))):.2f}"],
        ]
        
        if 'mean_uncertainty' in stats:
            summary_data.append(["Mean Uncertainty", f"{stats['mean_uncertainty']:.2f}"])
            summary_data.append(["Uncertainty Ratio", f"{stats.get('uncertainty_ratio', 0):.3f}"])
        
        for label, value in summary_data:
            print(f"{label:30s}: {value}")
        
        print("=" * 80)
        
        # Print first few forecast values
        forecast_dates = forecast_results.get("forecast_dates", forecast_results.get("date", []))
        point_forecasts = forecast_results.get("point_forecasts", forecast_results.get("forecast", []))
        
        if len(forecast_dates) > 0 and len(point_forecasts) > 0:
            print("\nFirst 10 Forecast Values:")
            print("-" * 40)
            print(f"{'Date':15s} {'Point Forecast':15s}")
            
            if 'lower_bounds' in forecast_results and 'upper_bounds' in forecast_results:
                print(f"{'Date':15s} {'Point Forecast':15s} {'Lower Bound':15s} {'Upper Bound':15s}")
                print("-" * 40)
                
                for i in range(min(10, len(forecast_dates))):
                    date_str = forecast_dates[i].strftime('%Y-%m-%d') \
                              if hasattr(forecast_dates[i], 'strftime') \
                              else str(forecast_dates[i])
                    
                    print(f"{date_str:15s} "
                          f"{point_forecasts[i]:15.2f} "
                          f"{forecast_results['lower_bounds'][i]:15.2f} "
                          f"{forecast_results['upper_bounds'][i]:15.2f}")
            else:
                print("-" * 40)
                for i in range(min(10, len(forecast_dates))):
                    date_str = forecast_dates[i].strftime('%Y-%m-%d') \
                              if hasattr(forecast_dates[i], 'strftime') \
                              else str(forecast_dates[i])
                    print(f"{date_str:15s} {point_forecasts[i]:15.2f}")
    
    def create_forecast_dashboard(
        self,
        forecast_df: pd.DataFrame,
        historical_df: Optional[pd.DataFrame] = None,
        actual_df: Optional[pd.DataFrame] = None,
        title: str = "Forecast Analytics Dashboard",
        include_components: bool = True,
        include_errors: bool = True
    ) -> Dict[str, go.Figure]:
        """
        Create comprehensive forecast dashboard with multiple plots.
        
        Args:
            forecast_df: DataFrame with forecast results
            historical_df: Historical data
            actual_df: Actual values for comparison (if available)
            title: Dashboard title
            include_components: Whether to include component plots
            include_errors: Whether to include error analysis
            
        Returns:
            Dictionary of plotly figures
        """
        dashboard = {}
        
        # 1. Main forecast plot
        dashboard["timeseries"] = self.plot_forecast_timeseries(
            forecast_df,
            historical_df,
            title=f"{title} - Forecast"
        )
        
        # 2. Forecast distribution
        dashboard["distribution"] = self.plot_forecast_distribution_plotly(
            forecast_df,
            title="Forecast Value Distribution"
        )
        
        # 3. Uncertainty plot
        if "lower_bound" in forecast_df.columns and "upper_bound" in forecast_df.columns:
            dashboard["uncertainty"] = self.plot_forecast_uncertainty(
                forecast_df,
                title="Forecast Uncertainty"
            )
        
        # 4. Component plots
        if include_components:
            component_cols = [col for col in forecast_df.columns if any(
                keyword in col.lower() for keyword in ["trend", "seasonal", "cycle", "residual"]
            )]
            
            if component_cols:
                dashboard["components"] = self.plot_forecast_components(
                    forecast_df,
                    component_cols,
                    title="Forecast Components"
                )
        
        # 5. Error analysis
        if include_errors and actual_df is not None:
            dashboard["vs_actual"] = self.plot_forecast_vs_actual(
                forecast_df,
                actual_df,
                title="Forecast vs Actual"
            )
            
            dashboard["error_analysis"] = self.plot_forecast_error_analysis(
                forecast_df,
                actual_df,
                title="Error Analysis"
            )
        
        # 6. Waterfall chart
        if "date" in forecast_df.columns:
            start_value = historical_df["unit_sales"].iloc[-1] if historical_df is not None else 0
            dashboard["waterfall"] = self.plot_forecast_waterfall(
                forecast_df,
                start_value,
                title="Forecast Changes Waterfall"
            )
        
        return dashboard
    
    # Add after other imports

# Add helper function
def create_executive_dashboard(forecast_df: pd.DataFrame, 
                              historical_df: pd.DataFrame,
                              config: Dict[str, Any]) -> None:
    """
    Quick function to create executive dashboard.
    
    Args:
        forecast_df: Forecast results DataFrame
        historical_df: Historical data DataFrame
        config: Forecast configuration
    """
    # Lazy import to avoid circular dependency
    from app.components.executive_dashboard import ExecutiveDashboard
    ExecutiveDashboard.create_kpi_dashboard(forecast_df, historical_df, config)
 