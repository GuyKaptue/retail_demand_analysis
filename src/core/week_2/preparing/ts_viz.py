# src/core/week_2/models/preparing/ts_viz.py

import matplotlib.pyplot as plt  # type: ignore
import seaborn as sns  # type: ignore
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf  # type: ignore
import numpy as np  # type: ignore
import pandas as pd  # type: ignore
from darts import TimeSeries  # type: ignore
from darts.models import ARIMA  # type: ignore
import os
from typing import Optional, Tuple, List, Dict, Any, Union  # noqa: F401
from scipy import stats  # type: ignore
import warnings
warnings.filterwarnings('ignore')

from src.utils import get_path  # Import the path utility  # noqa: E402




class TSVisualization:
    """
    Comprehensive visualization toolkit for time series models (ARIMA, SARIMA, etc.).
    Supports both dark and Heller-inspired emotional palettes.
    """

    # Class-level constants for figure size
    FIGSIZE = (18, 12)

    # Deep analytical palette (original dark theme)
    DARK_PALETTE = {
        'primary':   '#0a1929',   # Very dark navy
        'secondary': '#0d2d4a',   # Dark blue
        'tertiary':  '#1a2332',   # Dark slate
        'line1':     '#1c3a57',   # Deep blue
        'line2':     '#243447',   # Dark gray-blue
        'line3':     '#0f2942',   # Navy blue
        'accent':    '#16213e',   # Accent navy
        'black':     '#000000',   # Pure black
    }

    # Heller-inspired emotional palette (bright, expressive)
    HELLER_PALETTE = {
        'yellow':  '#FFD300',   # Optimism, clarity, energy
        'orange':  '#FF8C00',   # Warmth, stimulation, appetite
        'red':     '#E10600',   # Passion, urgency, intensity
        'pink':    '#FF5DA2',   # Softness, charm, friendliness
        'green':   '#00A86B',   # Balance, growth, calm
        'blue':    '#1E90FF',   # Trust, clarity, openness
        'purple':  '#8A2BE2',   # Creativity, imagination
        'white':   '#FFFFFF',   # Clean contrast for dark backgrounds
    }

    def __init__(self, save_dir: str, week: int = 2, style: str = "seaborn-v0_8", palette: str = "heller"):
        """
        Initialize visualization toolkit.

        Args:
            week: Week number for path resolution
            style: Matplotlib style (default: seaborn-v0_8)
            palette: Color palette to use ('dark' or 'heller')
        """
        self.week = week
        self.style = style
        self.palette = palette
        self.save_dir = save_dir if save_dir is not None else get_path("arima_viz")

        # Set global plotting style
        plt.style.use(style)
        sns.set_palette("dark" if palette == "dark" else "bright")

        # Create subdirectories for better organization
        self.subdirs = ["forecasts", "residuals", "diagnostics", "comparisons", "decomposition"]
        for subdir in self.subdirs:
            os.makedirs(os.path.join(self.save_dir, subdir), exist_ok=True)

        print(f"[TimeSeriesViz] Visualizations will be saved to: {self.save_dir}")
        print(f"[TimeSeriesViz] Style: {style}, Palette: {palette}")

    @property
    def palette(self):
        return self._palette

    @palette.setter
    def palette(self, value):
        self._palette = value
        if value == "dark":
            self._current_palette = self.DARK_PALETTE
        elif value == "heller":
            self._current_palette = self.HELLER_PALETTE
        else:
            raise ValueError("Palette must be 'dark' or 'heller'")

    # =========================================================================
    # 0. SAVE PLOT METHOD (NEW)
    # =========================================================================

    def save_plot(
        self, 
        filename: str, 
        subfolder: str = None, 
        dpi: int = 300, 
        bbox_inches: str = 'tight', 
        pad_inches: float = 0.1
    ) -> str:
        """
        Save the current matplotlib figure to the appropriate subfolder.
        
        Args:
            filename: Name of the file to save (including extension)
            subfolder: Subfolder to save in (one of: 'forecasts', 'residuals', 
                     'diagnostics', 'comparisons', 'decomposition'). If None,
                     saves directly to save_dir
            dpi: Resolution in dots per inch
            bbox_inches: Bounding box in inches
            pad_inches: Padding in inches
            
        Returns:
            str: Full path where the plot was saved
        """
        # Validate subfolder
        if subfolder and subfolder not in self.subdirs:
            valid_folders = ", ".join(self.subdirs)
            print(f"⚠ Warning: subfolder '{subfolder}' not in predefined list. "
                  f"Valid options: {valid_folders}. Creating custom subfolder.")
            custom_dir = os.path.join(self.save_dir, subfolder)
            os.makedirs(custom_dir, exist_ok=True)
            save_dir = custom_dir
        elif subfolder:
            save_dir = os.path.join(self.save_dir, subfolder)
        else:
            save_dir = self.save_dir
        
        # Create full path
        save_path = os.path.join(save_dir, filename)
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Save the plot
        plt.savefig(
            save_path, 
            dpi=dpi, 
            bbox_inches=bbox_inches,
            pad_inches=pad_inches
        )
        
        print(f"💾 Plot saved to: {save_path}")
        return save_path

    # =========================================================================
    # 1. FORECAST VISUALIZATIONS
    # =========================================================================

    def forecast_overlay(
        self,
        train: TimeSeries,
        test: TimeSeries,
        forecast: TimeSeries,
        title: str = "Time Series Forecast",
        confidence_intervals: Optional[Tuple[TimeSeries, TimeSeries]] = None,
        filename: str = "forecast_overlay.png",
        show_train: bool = True,
        show_test: bool = True,
       
    ) -> None:
        """Enhanced forecast visualization with selected palette."""
        print("[TimeSeriesViz] Generating enhanced forecast overlay...")
       
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.FIGSIZE, gridspec_kw={'height_ratios': [3, 1]})

        # Map dark palette keys to Heller palette keys
        if self.palette == "heller":
            train_color = self._current_palette['orange']  # Example mapping
            test_color = self._current_palette['red']      # Example mapping
            forecast_color = self._current_palette['blue'] # Example mapping
            ci_color = self._current_palette['purple']     # Example mapping
        else:
            train_color = self._current_palette['secondary']
            test_color = self._current_palette['line1']
            forecast_color = self._current_palette['primary']
            ci_color = self._current_palette['primary']

        # Main forecast plot
        if show_train:
            train.plot(ax=ax1, label="Training Data", color=train_color, alpha=0.7, linewidth=2)

        if show_test:
            test.plot(ax=ax1, label="Actual Test Data", color=test_color, alpha=0.9, linewidth=2.5)

        forecast.plot(ax=ax1, label="Forecast", color=forecast_color, linewidth=2.5, marker='o', markersize=4)

        # Add confidence intervals if provided
        if confidence_intervals:
            lower, upper = confidence_intervals
            ax1.fill_between(
                forecast.time_index,
                lower.values().flatten(),
                upper.values().flatten(),
                color=ci_color,
                alpha=0.2,
                label='95% Confidence Interval'
            )

        ax1.set_title(title, fontsize=16, fontweight='bold')
        ax1.set_ylabel("Value", fontsize=12)
        ax1.legend(loc='best', fontsize=10)
        ax1.grid(True, alpha=0.3)

        # Add zoomed-in view of forecast period
        if show_test:
            forecast_period = test.time_index
            forecast_vals = forecast.values().flatten()
            test_vals = test.values().flatten()

            ax2.plot(forecast_period, test_vals, color=test_color, label='Actual', linewidth=2.5, marker='s', markersize=5)
            ax2.plot(forecast_period, forecast_vals, color=forecast_color, label='Forecast', linewidth=2.5, marker='o', markersize=5)
            ax2.fill_between(
                forecast_period,
                test_vals - np.abs(forecast_vals - test_vals),
                test_vals + np.abs(forecast_vals - test_vals),
                alpha=0.3,
                color=self._current_palette['green'] if self.palette == "heller" else self._current_palette['tertiary'],
                label='Error Range'
            )

            ax2.set_title("Forecast vs Actual (Zoomed)", fontsize=14)
            ax2.set_xlabel("Time")
            ax2.set_ylabel("Value")
            ax2.legend(loc='best')
            ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save the plot using the new method
        self.save_plot(filename, subfolder="forecasts")
        plt.show()


    def multi_step_forecast(
        self,
        historical: TimeSeries,
        forecasts: Dict[str, TimeSeries],
        title: str = "Multi-Step Forecast Comparison",
        filename: str = "multi_step_forecast.png",
        save_dir: Optional[str] = None
    ) -> None:
        """Compare multiple forecast horizons or models with selected palette."""
        print("[TimeSeriesViz] Generating multi-step forecast comparison...")
        if save_dir is not None:
            self.save_dir = save_dir
            
        plt.figure(figsize=self.FIGSIZE)

        # Plot historical data
        historical.plot(label="Historical Data", color=self._current_palette['black'], alpha=0.8, linewidth=2.5)

        # Plot each forecast with palette colors
        colors = list(self._current_palette.values())[:len(forecasts)]

        for idx, (label, forecast) in enumerate(forecasts.items()):
            color = colors[idx % len(colors)]
            forecast.plot(label=f"{label} Forecast", color=color, linewidth=2.5, marker='o', markersize=4)

        plt.title(title, fontsize=16, fontweight='bold')
        plt.xlabel("Time", fontsize=12)
        plt.ylabel("Value", fontsize=12)
        plt.legend(loc='best', fontsize=10)
        plt.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save using the new method
        self.save_plot(filename, subfolder="forecasts")
        plt.show()


    # =========================================================================
    # 2. RESIDUAL ANALYSIS & DIAGNOSTICS
    # =========================================================================
    
    def residual_diagnostics(
        self,
        residuals: np.ndarray,
        title_prefix: str = "Model Residuals",
        filename_prefix: str = "residual_diagnostics",
        lags: int = 40,
        save_dir: Optional[str] = None
    ) -> None:
        """Comprehensive residual diagnostics with selected palette."""
        print("[TimeSeriesViz] Generating residual diagnostics...")
        if save_dir is not None:
            self.save_dir = save_dir
        # Map dark palette keys to Heller palette keys
        if self.palette == "heller":
            secondary_color = self._current_palette['green']
            primary_color = self._current_palette['blue']
            tertiary_color = self._current_palette['orange']
            black_color = self._current_palette['purple']
        else:
            secondary_color = self._current_palette['secondary']
            primary_color = self._current_palette['primary']
            tertiary_color = self._current_palette['tertiary']
            black_color = self._current_palette['black']

        fig, axes = plt.subplots(2, 2, figsize=self.FIGSIZE)
        fig.suptitle(f"{title_prefix} - Diagnostic Plots", fontsize=16, fontweight='bold')

        # 1. Residual time series plot
        axes[0, 0].plot(residuals, 'o-', alpha=0.7, markersize=3, color=secondary_color)
        axes[0, 0].axhline(y=0, color=primary_color, linestyle='--', alpha=0.7, linewidth=2)
        axes[0, 0].set_title("Residuals Over Time", fontweight='bold')
        axes[0, 0].set_xlabel("Observation")
        axes[0, 0].set_ylabel("Residual")
        axes[0, 0].grid(True, alpha=0.3)

        # 2. Histogram with normal distribution overlay
        axes[0, 1].hist(residuals, bins=30, density=True, alpha=0.7, edgecolor=black_color, color=tertiary_color)

        # Add normal distribution overlay
        mu, std = np.mean(residuals), np.std(residuals)
        xmin, xmax = axes[0, 1].get_xlim()
        x = np.linspace(xmin, xmax, 100)
        p = stats.norm.pdf(x, mu, std)
        axes[0, 1].plot(x, p, color=primary_color, linewidth=2.5, label=f'N({mu:.2f}, {std:.2f})')
        axes[0, 1].set_title("Distribution of Residuals", fontweight='bold')
        axes[0, 1].set_xlabel("Residual Value")
        axes[0, 1].set_ylabel("Density")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # 3. Q-Q plot
        stats.probplot(residuals, dist="norm", plot=axes[1, 0])
        axes[1, 0].set_title("Q-Q Plot", fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].get_lines()[0].set_color(secondary_color)
        axes[1, 0].get_lines()[1].set_color(primary_color)

        # 4. ACF of residuals
        plot_acf(residuals, lags=lags, ax=axes[1, 1], alpha=0.05, color=secondary_color)
        axes[1, 1].set_title(f"ACF of Residuals (lags={lags})", fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        # Save the plot using the new method
        self.save_plot(f"{filename_prefix}_diagnostics.png", subfolder="residuals")
        plt.show()


    def residual_acf_pacf(
        self,
        residuals: np.ndarray,
        lags: int = 30,
        title: str = "Residual ACF & PACF",
        filename: str = "residual_acf_pacf.png"
    ) -> None:
        """Combined ACF and PACF plot for residuals with selected palette."""
        print("[TimeSeriesViz] Generating residual ACF/PACF...")

        # Map dark palette keys to Heller palette keys
        if self.palette == "heller":
            secondary_color = self._current_palette['green']
            primary_color = self._current_palette['blue']
        else:
            secondary_color = self._current_palette['secondary']
            primary_color = self._current_palette['primary']

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.FIGSIZE)

        # ACF plot
        plot_acf(residuals, lags=lags, ax=ax1, alpha=0.05, color=secondary_color)
        ax1.set_title("Autocorrelation Function (ACF)", fontweight='bold')
        ax1.set_xlabel("Lag")
        ax1.set_ylabel("Correlation")
        ax1.grid(True, alpha=0.3)

        # PACF plot
        plot_pacf(residuals, lags=lags, ax=ax2, alpha=0.05, method='ywm', color=primary_color)
        ax2.set_title("Partial Autocorrelation Function (PACF)", fontweight='bold')
        ax2.set_xlabel("Lag")
        ax2.set_ylabel("Partial Correlation")
        ax2.grid(True, alpha=0.3)

        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()

        # Save using the new method
        self.save_plot(filename, subfolder="residuals")
        plt.show()


    # =========================================================================
    # 3. TIME SERIES DECOMPOSITION
    # =========================================================================

    def ts_to_dataframe(self, series):
        """Convert a Darts TimeSeries to a pandas DataFrame with datetime index."""
        try:
            if hasattr(series, "to_pandas"):
                df = series.to_pandas()
                if isinstance(df, pd.Series):
                    df = df.to_frame(name="value")
                return df
            elif hasattr(series, "pd_dataframe"):
                return series.pd_dataframe()
            else:
                return pd.DataFrame(series.values(), index=series.time_index, columns=["value"])
        except Exception as e:
            raise RuntimeError(f"Conversion failed: {e}")

    def decompose_series(
        self,
        series: TimeSeries,
        period: int = 7,
        model: str = 'additive',
        title: str = "Time Series Decomposition",
        filename: str = "series_decomposition.png"
    ) -> None:
        """Decompose time series with selected palette."""
        print("[TimeSeriesViz] Decomposing time series...")

        # Convert to pandas for decomposition
        df = self.ts_to_dataframe(series)

        from statsmodels.tsa.seasonal import seasonal_decompose

        try:
            decomposition = seasonal_decompose(df.iloc[:, 0], model=model, period=period)

            fig, axes = plt.subplots(4, 1, figsize=self.FIGSIZE)
            fig.suptitle(f"{title} - {model.capitalize()} Model (period={period})", fontsize=16, fontweight='bold')

            colors = list(self._current_palette.values())[:4]

            # Original series
            axes[0].plot(decomposition.observed, color=colors[0], linewidth=2)
            axes[0].set_ylabel("Observed", fontweight='bold')
            axes[0].grid(True, alpha=0.3)

            # Trend component
            axes[1].plot(decomposition.trend, color=colors[1], linewidth=2)
            axes[1].set_ylabel("Trend", fontweight='bold')
            axes[1].grid(True, alpha=0.3)

            # Seasonal component
            axes[2].plot(decomposition.seasonal, color=colors[2], linewidth=2)
            axes[2].set_ylabel("Seasonal", fontweight='bold')
            axes[2].grid(True, alpha=0.3)

            # Residual component
            axes[3].plot(decomposition.resid, 'o-', markersize=2, color=colors[3])
            axes[3].axhline(y=0, color=self._current_palette['black'], linestyle='--', alpha=0.5)
            axes[3].set_ylabel("Residual", fontweight='bold')
            axes[3].set_xlabel("Time")
            axes[3].grid(True, alpha=0.3)

            plt.tight_layout()

            # Save using the new method
            self.save_plot(filename, subfolder="decomposition")
            plt.show()

            return decomposition

        except Exception as e:
            print(f"⚠ Decomposition failed: {e}")
            return None

    def rolling_statistics(
        self,
        series: TimeSeries,
        windows: List[int] = [7, 30, 90],
        statistics: List[str] = ['mean', 'std'],
        title: str = "Rolling Statistics",
        filename: str = "rolling_statistics.png"
    ) -> None:
        """Plot rolling statistics with selected palette."""
        print("[TimeSeriesViz] Generating rolling statistics...")

        df = self.ts_to_dataframe(series)
        col_name = df.columns[0]

        n_stats = len(statistics)
        n_windows = len(windows)
        fig, axes = plt.subplots(n_stats, n_windows, figsize=self.FIGSIZE)

        if n_stats == 1 and n_windows == 1:
            axes = np.array([[axes]])
        elif n_stats == 1:
            axes = axes.reshape(1, -1)
        elif n_windows == 1:
            axes = axes.reshape(-1, 1)

        # Map dark palette keys to Heller palette keys
        if self.palette == "heller":
            original_color = self._current_palette['green']  # Example mapping for 'tertiary'
            line_colors = [self._current_palette['blue'], self._current_palette['orange'], self._current_palette['red']]
        else:
            original_color = self._current_palette['tertiary']
            line_colors = [self._current_palette['primary'], self._current_palette['secondary'], self._current_palette['line1']]

        for i, stat in enumerate(statistics):
            for j, window in enumerate(windows):
                ax = axes[i, j]

                # Compute rolling statistic
                if stat == 'mean':
                    rolling = df[col_name].rolling(window=window, center=True).mean()
                elif stat == 'std':
                    rolling = df[col_name].rolling(window=window, center=True).std()
                elif stat == 'min':
                    rolling = df[col_name].rolling(window=window, center=True).min()
                elif stat == 'max':
                    rolling = df[col_name].rolling(window=window, center=True).max()
                elif stat == 'median':
                    rolling = df[col_name].rolling(window=window, center=True).median()
                else:
                    continue

                # Plot
                ax.plot(df.index, df[col_name], alpha=0.3, label='Original', linewidth=1, color=original_color)
                ax.plot(rolling.index, rolling.values, label=f'{window}-day {stat}', linewidth=2.5, color=line_colors[j % len(line_colors)])
                ax.set_title(f'{stat.capitalize()} (window={window})', fontweight='bold')
                ax.set_xlabel("Date")
                ax.set_ylabel("Value")
                ax.legend(fontsize=9)
                ax.grid(True, alpha=0.3)

        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()

        # Save using the new method
        self.save_plot(filename, subfolder="diagnostics")
        plt.show()


    # =========================================================================
    # 4. MODEL COMPARISON VISUALIZATIONS
    # =========================================================================

    def model_comparison(
        self,
        test: TimeSeries,
        forecasts: Dict[str, TimeSeries],
        metrics: Optional[Dict[str, Dict[str, float]]] = None,
        title: str = "Model Comparison",
        filename: str = "model_comparison.png"
    ) -> None:
        """Compare multiple models with selected palette."""
        print("[TimeSeriesViz] Generating model comparison...")

        fig, axes = plt.subplots(2, 2, figsize=self.FIGSIZE)
        fig.suptitle(title, fontsize=18, fontweight='bold')

        # Convert to arrays
        test_vals = test.values().flatten()
        test_dates = test.time_index

        # Palette colors for models
        colors = list(self._current_palette.values())[:len(forecasts)]

        # 1. All forecasts overlay
        ax1 = axes[0, 0]
        test.plot(ax=ax1, label="Actual", color=self._current_palette['black'], linewidth=3)

        for idx, (model_name, forecast) in enumerate(forecasts.items()):
            color = colors[idx % len(colors)]
            forecast.plot(ax=ax1, label=model_name, color=color, linewidth=2.5, alpha=0.8)

        ax1.set_title("All Forecasts Overlay", fontweight='bold')
        ax1.set_xlabel("Time")
        ax1.set_ylabel("Value")
        ax1.legend(loc='best', fontsize=9)
        ax1.grid(True, alpha=0.3)

        # 2. Forecast errors
        ax2 = axes[0, 1]
        for idx, (model_name, forecast) in enumerate(forecasts.items()):
            color = colors[idx % len(colors)]
            forecast_vals = forecast.values().flatten()
            errors = forecast_vals - test_vals
            ax2.plot(test_dates, errors, 'o-', color=color, label=model_name, markersize=3, alpha=0.7, linewidth=2)

        ax2.axhline(y=0, color=self._current_palette['black'], linestyle='--', alpha=0.5, linewidth=2)
        ax2.set_title("Forecast Errors", fontweight='bold')
        ax2.set_xlabel("Time")
        ax2.set_ylabel("Error (Predicted - Actual)")
        ax2.legend(loc='best', fontsize=9)
        ax2.grid(True, alpha=0.3)

        # 3. Error distribution
        ax3 = axes[1, 0]
        error_data = []
        labels = []
        for model_name, forecast in forecasts.items():
            forecast_vals = forecast.values().flatten()
            errors = forecast_vals - test_vals
            error_data.append(errors)
            labels.append(model_name)

        bp = ax3.boxplot(error_data, labels=labels, patch_artist=True)
        for patch, color in zip(bp['boxes'], colors[:len(error_data)]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax3.axhline(y=0, color=self._current_palette['black'], linestyle='--', alpha=0.5)
        ax3.set_title("Error Distribution by Model", fontweight='bold')
        ax3.set_ylabel("Error")
        ax3.grid(True, alpha=0.3, axis='y')

        # 4. Metrics comparison
        ax4 = axes[1, 1]
        if metrics:
            metric_names = list(next(iter(metrics.values())).keys())
            n_metrics = len(metric_names)
            n_models = len(metrics)

            x = np.arange(n_metrics)
            width = 0.8 / n_models

            for i, (model_name, model_metrics) in enumerate(metrics.items()):
                color = colors[i % len(colors)]
                values = [model_metrics[metric] for metric in metric_names]
                ax4.bar(x + i*width - width*(n_models-1)/2, values, width, label=model_name, alpha=0.8, color=color)

            ax4.set_xlabel("Metric")
            ax4.set_ylabel("Value")
            ax4.set_title("Performance Metrics Comparison", fontweight='bold')
            ax4.set_xticks(x)
            ax4.set_xticklabels(metric_names, rotation=45, ha='right')
            ax4.legend(loc='best', fontsize=9)
            ax4.grid(True, alpha=0.3, axis='y')
        else:
            ax4.text(0.5, 0.5, "No metrics provided", ha='center', va='center', transform=ax4.transAxes, fontsize=12)
            ax4.set_title("Metrics Comparison", fontweight='bold')
            ax4.axis('off')

        plt.tight_layout()

        # Save using the new method
        self.save_plot(filename, subfolder="comparisons")
        plt.show()


    def accuracy_metrics_chart(
        self,
        metrics_dict: Dict[str, Dict[str, float]],
        title: str = "Model Accuracy Metrics",
        filename: str = "accuracy_metrics_chart.png"
    ) -> None:
        """Create a radar chart with selected palette."""
        print("[TimeSeriesViz] Generating accuracy metrics chart...")

        models = list(metrics_dict.keys())
        metrics = list(next(iter(metrics_dict.values())).keys())
        n_metrics = len(metrics)

        # Normalize metrics
        normalized_data = {}
        for model_name, model_metrics in metrics_dict.items():
            normalized = {}
            for metric, value in model_metrics.items():
                all_values = [m[metric] for m in metrics_dict.values()]
                if any('error' in metric.lower() or 'mae' in metric.lower() or 'rmse' in metric.lower() or 'mse' in metric.lower() for metric in metrics):
                    normalized[metric] = 1 - ((value - min(all_values)) / (max(all_values) - min(all_values) + 1e-10))
                else:
                    normalized[metric] = (value - min(all_values)) / (max(all_values) - min(all_values) + 1e-10)
            normalized_data[model_name] = normalized

        # Create radar chart
        angles = np.linspace(0, 2*np.pi, n_metrics, endpoint=False).tolist()
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=self.FIGSIZE, subplot_kw=dict(projection='polar'))

        colors = list(self._current_palette.values())[:len(models)]

        for idx, (model_name, normalized) in enumerate(normalized_data.items()):
            color = colors[idx % len(colors)]
            values = [normalized[metric] for metric in metrics]
            values += values[:1]

            ax.plot(angles, values, 'o-', linewidth=2.5, label=model_name, color=color)
            ax.fill(angles, values, alpha=0.2, color=color)

        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics, fontsize=10)
        ax.set_rlabel_position(0)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], color="gray", fontsize=8)
        ax.set_ylim(0, 1)

        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        plt.title(title, fontsize=16, fontweight='bold', pad=20)

        plt.tight_layout()

        # Save using the new method
        self.save_plot(filename, subfolder="comparisons")
        plt.show()


    # =========================================================================
    # 5. UTILITY VISUALIZATIONS
    # =========================================================================

    def correlation_heatmap(
        self,
        df: pd.DataFrame,
        title: str = "Feature Correlation Heatmap",
        filename: str = "correlation_heatmap.png"
    ) -> None:
        """Create correlation heatmap with selected palette."""
        print("[TimeSeriesViz] Generating correlation heatmap...")

        corr_matrix = df.corr()

        plt.figure(figsize=self.FIGSIZE)

        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

        # Plot heatmap
        sns.heatmap(
            corr_matrix,
            mask=mask,
            annot=True,
            fmt=".2f",
            cmap="Blues" if self.palette == "dark" else "viridis",
            center=0,
            square=True,
            linewidths=0.5,
            cbar_kws={"shrink": 0.8}
        )

        plt.title(title, fontsize=16, fontweight='bold')
        plt.tight_layout()

        # Save using the new method
        self.save_plot(filename, subfolder="diagnostics")
        plt.show()


    def prediction_intervals(
        self,
        historical: TimeSeries,
        predictions: List[TimeSeries],
        intervals: List[float] = [0.5, 0.8, 0.95],
        title: str = "Prediction Intervals",
        filename: str = "prediction_intervals.png"
    ) -> None:
        """Visualize prediction intervals at different confidence levels."""
        print("[TimeSeriesViz] Generating prediction intervals...")

        if not predictions:
            print("⚠ No predictions provided for intervals")
            return

        pred_array = np.array([pred.values().flatten() for pred in predictions])

        plt.figure(figsize=self.FIGSIZE)

        # Plot historical data
        historical.plot(label="Historical Data", color=self._current_palette['black'], alpha=0.8, linewidth=2.5)

        # Calculate and plot intervals
        colors = list(self._current_palette.values())[2:5]

        sorted_intervals = sorted(intervals, reverse=True)

        for idx, interval in enumerate(sorted_intervals):
            color = colors[idx % len(colors)]

            lower = np.percentile(pred_array, (1 - interval) * 50, axis=0)
            upper = np.percentile(pred_array, (1 + interval) * 50, axis=0)

            plt.fill_between(
                predictions[0].time_index,
                lower,
                upper,
                color=color,
                alpha=0.4,
                label=f'{int(interval*100)}% Confidence'
            )

        # Plot mean prediction
        mean_pred = np.mean(pred_array, axis=0)
        plt.plot(
            predictions[0].time_index,
            mean_pred,
            color=self._current_palette['primary'],
            linewidth=2.5,
            label='Mean Prediction'
        )

        plt.title(title, fontsize=16, fontweight='bold')
        plt.xlabel("Time", fontsize=12)
        plt.ylabel("Value", fontsize=12)
        plt.legend(loc='best', fontsize=10)
        plt.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save using the new method
        self.save_plot(filename, subfolder="forecasts")
        plt.show()


    # =========================================================================
    # 6. LEGACY METHODS (for backward compatibility)
    # =========================================================================

    def overlay_forecast(
        self,
        train: TimeSeries,
        test: TimeSeries,
        forecast: TimeSeries,
        title: str,
        filename: str = "arima_forecast_overlay.png"
    ) -> None:
        self.forecast_overlay(train, test, forecast, title, filename=filename)

    def residual_acf(
        self,
        model: 'ARIMA',
        train_ts: TimeSeries,
        lags: int = 30,
        title: str = "Residual ACF",
        filename: str = "arima_residual_acf.png"
    ) -> None:
        fitted = model.predict(len(train_ts), series=train_ts)
        residuals = train_ts.values().flatten() - fitted.values().flatten()

        fig, ax = plt.subplots(figsize=self.FIGSIZE)
        plot_acf(residuals, lags=lags, ax=ax, alpha=0.05, color=self._current_palette['secondary'])
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        # Save using the new method
        self.save_plot(filename, subfolder="residuals")
        plt.show()


    def rolling_mean_plot(
        self,
        df_calendar: pd.DataFrame,
        sales_col: str = "unit_sales",
        window: int = 30,
        title: str = "30-day rolling mean",
        filename: str = "rolling_mean_sales.png"
    ) -> None:
        print("[TimeSeriesViz] Generating rolling mean plot...")

        rolling = df_calendar[sales_col].rolling(window=window, center=True).mean()

        plt.figure(figsize=self.FIGSIZE)
        plt.plot(
            df_calendar.index,
            df_calendar[sales_col],
            color=self._current_palette['tertiary'],
            alpha=0.5,
            linewidth=1,
            label='Daily unit_sales'
        )
        plt.plot(
            rolling.index,
            rolling.values,
            color=self._current_palette['primary'],
            linewidth=2.5,
            label=f'{window}-day rolling mean'
        )

        plt.title(title, fontsize=16, fontweight='bold')
        plt.xlabel("Date")
        plt.ylabel("Unit Sales")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        # Save using the new method
        self.save_plot(filename)
        plt.show()


    # =========================================================================
    # 7. HELPER METHODS
    # =========================================================================

    def set_style(self, style: str):
        """Change plotting style dynamically."""
        self.style = style
        plt.style.use(style)
        sns.set_palette("husl")
        print(f"[TimeSeriesViz] Style changed to: {style}")

    def get_save_path(self, subcategory: str, filename: str) -> str:
        """Get full save path for a visualization."""
        return os.path.join(self.save_dir, subcategory, filename)

    def create_summary_report(
        self,
        visualizations: List[str],
        title: str = "Model Analysis Report",
        filename: str = "analysis_report.txt"
    ) -> None:
        """
        Create a text summary report of all visualizations.

        Args:
            visualizations: List of visualization filenames
            title: Report title
            filename: Output filename
        """
        report_path = os.path.join(self.save_dir, filename)

        with open(report_path, 'w') as f:
            f.write(f"{'='*60}\n")
            f.write(f"{title}\n")
            f.write(f"{'='*60}\n\n")
            f.write(f"Generated: {pd.Timestamp.now()}\n")
            f.write(f"Week: {self.week}\n")
            f.write(f"Style: {self.style}\n")
            f.write(f"Palette: {self.palette}\n\n")

            f.write("Visualizations Generated:\n")
            f.write("-"*40 + "\n")
            for viz in visualizations:
                f.write(f"• {viz}\n")

            f.write(f"\nSave Directory: {self.save_dir}\n")
            f.write(f"\n{'='*60}\n")

        print(f"📄 Summary report saved to: {report_path}")

    @staticmethod
    def plot_filtered_sales(
        series: TimeSeries,
        title: str = "Filtered Daily Sales",
        figsize: tuple = FIGSIZE,
        save_path: str = None,
        color: str = '#0a1929'
    ) -> None:
        """Visualize a filtered sales time series with selected palette."""
        plt.figure(figsize=figsize)

        if isinstance(series, TimeSeries):
            df = series.to_pandas() if hasattr(series, "to_pandas") else pd.DataFrame(series.values(), index=series.time_index)
            df.plot(color=color, linewidth=2, legend=False)
        elif isinstance(series, (pd.Series, pd.DataFrame)):
            series.plot(color=color, linewidth=2, legend=False)
        else:
            raise TypeError("Input must be a Darts TimeSeries or pandas Series/DataFrame")

        plt.title(title, fontsize=16, fontweight="bold")
        plt.xlabel("Date", fontsize=12)
        plt.ylabel("Unit Sales", fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"💾 Plot saved to {save_path}")

        plt.show()

    
    def plot_daily_vs_monthly(
        self,
        series: Union[pd.DataFrame, pd.Series],
        value_col: str = "unit_sales",
        title: str = "Daily vs Monthly Unit Sales",
        figsize: Optional[Tuple[int, int]] = None,
        filename: Optional[str] = "daily_vs_monthly.png"
    ) -> None:
        """Plot daily and monthly aggregated time series with selected palette."""
        print("[TimeSeriesViz] Plotting daily vs monthly series...")

        if isinstance(series, pd.Series):
            df = series.to_frame(name=value_col)
        elif isinstance(series, pd.DataFrame):
            df = series.copy()
        else:
            raise TypeError("series must be pandas Series or DataFrame")

        monthly_df = df.resample("M").sum()

        fig, ax1 = plt.subplots(figsize=figsize or self.FIGSIZE)

        # Map dark palette keys to Heller palette keys
        if self.palette == "heller":
            daily_color = self._current_palette['green']
            monthly_color = self._current_palette['blue']
            black_color = self._current_palette['purple']
        else:
            daily_color = self._current_palette['tertiary']
            monthly_color = self._current_palette['primary']
            black_color = self._current_palette['black']

        # Daily with mapped color
        ax1.plot(df.index, df[value_col], color=daily_color, linewidth=1, alpha=0.6, label="Daily Unit Sales")
        ax1.set_ylabel("Daily Unit Sales", color=black_color, fontsize=11, fontweight='bold')
        ax1.tick_params(axis="y", labelcolor=black_color)

        # Monthly with mapped color
        ax2 = ax1.twinx()
        ax2.plot(
            monthly_df.index,
            monthly_df[value_col],
            color=monthly_color,
            linewidth=2.5,
            label="Monthly Unit Sales",
            marker='o',
            markersize=4
        )
        ax2.set_ylabel("Monthly Unit Sales", color=monthly_color, fontsize=11, fontweight='bold')
        ax2.tick_params(axis="y", labelcolor=monthly_color)

        ax1.set_title(title, fontsize=16, fontweight="bold")
        ax1.set_xlabel("Date", fontsize=12)
        ax1.grid(True, alpha=0.3)

        ax1.legend(loc="upper left", fontsize=9)
        ax2.legend(loc="upper right", fontsize=9)

        plt.tight_layout()

        if filename:
            # Save using the new method
            self.save_plot(filename, subfolder="diagnostics")
            plt.show()