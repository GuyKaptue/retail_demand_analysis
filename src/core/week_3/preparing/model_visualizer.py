# src/core/week_3/analysis/model_visualizer.py

from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np  # noqa: F401

from scipy import stats
from typing import Optional, List, Tuple, Dict, Any  # noqa: F401
import warnings
from datetime import datetime  # noqa: F401
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error


class ModelVisualizer:
    """
    Professional visualization class for retail demand forecasting with time series regression.
    
    Provides comprehensive plots for:
    - Before modelling (EDA)
    - After modelling (Evaluation & Diagnostics)
    - Multi-model comparisons
    
    Features:
    - Professional color palette (Black, Blue, Orange, Violet)
    - Time series specific visualizations
    - Statistical diagnostics
    - Comprehensive error analysis
    - Feature importance visualization
    - Residual analysis
    - Multi-model comparison capabilities
    """

    # Professional Color Palette
    COLORS = {
        'primary': '#0A1929',      # Black/Dark Blue
        'secondary': '#1E90FF',    # Blue
        'accent': '#FF8C00',       # Orange
        'highlight': '#8B00FF',    # Violet
        'success': '#28A745',      # Green
        'error': '#DC3545',        # Red
        'warning': '#FFC107',      # Yellow
        'neutral': '#6C757D'       # Gray
    }
    
    # Model display names
    VARIANT_DISPLAY_NAMES = {
        "linear_regression": "Baseline",
        "linear_regression_grid": "Grid Search",
        "linear_regression_random": "Random Search",
        "linear_regression_hyperopt": "Hyperopt"
    }
    
    PALETTE = [COLORS['primary'], COLORS['secondary'], COLORS['accent'], COLORS['highlight']]
    MODEL_COLORS = {
        "linear_regression": COLORS['primary'],
        "linear_regression_grid": COLORS['secondary'],
        "linear_regression_random": COLORS['accent'],
        "linear_regression_hyperopt": COLORS['highlight']
    }

    def __init__(
        self, 
        df: pd.DataFrame, 
        target: str = "unit_sales",
        y_true: Optional[pd.Series] = None,
        y_pred_baseline: Optional[pd.Series] = None,
        y_pred_grid: Optional[pd.Series] = None,
        y_pred_random: Optional[pd.Series] = None,
        y_pred_hyperopt: Optional[pd.Series] = None,
        dates: Optional[pd.Series] = None, 
        store_nbr: Optional[pd.Series] = None,
        item_nbr: Optional[pd.Series] = None, 
        viz_path: Optional[str] = None
    ):
        """
        Initialize the ModelVisualizer with multiple model predictions.
        
        Args:
            df: Original dataframe
            target: Target variable name
            y_true: True values for evaluation
            y_pred_baseline: Baseline model predictions
            y_pred_grid: Grid Search model predictions
            y_pred_random: Random Search model predictions
            y_pred_hyperopt: Hyperopt model predictions
            dates: Date series for time series plots
            store_nbr: Store numbers for filtering
            item_nbr: Item numbers for filtering
            viz_path: Path to save visualizations
        """
        self.df = df
        self.target = target
        self.y_true = y_true
        self.dates = pd.to_datetime(dates) if dates is not None else None
        self.store_nbr = store_nbr
        self.item_nbr = item_nbr
        self.figsize = (14, 10)
        
        # Store all model predictions
        self.model_predictions = {}
        self.model_metrics = {}
        
        # Add provided predictions
        if y_pred_baseline is not None:
            self.model_predictions["linear_regression"] = y_pred_baseline
        if y_pred_grid is not None:
            self.model_predictions["linear_regression_grid"] = y_pred_grid
        if y_pred_random is not None:
            self.model_predictions["linear_regression_random"] = y_pred_random
        if y_pred_hyperopt is not None:
            self.model_predictions["linear_regression_hyperopt"] = y_pred_hyperopt
        
        # Calculate metrics for each model
        if y_true is not None:
            self._calculate_all_metrics()

        # Ensure viz_path is a Path object
        self.viz_path = Path(viz_path) if viz_path is not None else None
        if self.viz_path is not None:
            self.viz_path.mkdir(parents=True, exist_ok=True)
        
        # Setup plotting style
        self._setup_style()

    def _setup_style(self):
        """Setup consistent plotting style."""
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette(self.PALETTE)
        
        # Set default parameters
        plt.rcParams['figure.facecolor'] = 'white'
        plt.rcParams['axes.facecolor'] = '#F8F9FA'
        plt.rcParams['axes.edgecolor'] = self.COLORS['primary']
        plt.rcParams['axes.labelcolor'] = self.COLORS['primary']
        plt.rcParams['xtick.color'] = self.COLORS['primary']
        plt.rcParams['ytick.color'] = self.COLORS['primary']
        plt.rcParams['text.color'] = self.COLORS['primary']
        plt.rcParams['grid.alpha'] = 0.3

    def _calculate_metrics(self, y_pred, model_name: str):
        """Calculate metrics for a specific model."""
        if self.y_true is None:
            print("Debug: self.y_true is None")
            return None

        y_true = self.y_true
        y_pred_series = y_pred

        # Debug prints
        print(f"Debug: Type of y_true: {type(y_true)}")
        print(f"Debug: Type of y_pred_series: {type(y_pred_series)}")
        print(f"Debug: Length of y_true: {len(y_true)}")
        print(f"Debug: Length of y_pred_series: {len(y_pred_series)}")

        # Convert to pandas Series if not already
        if not isinstance(y_true, pd.Series):
            print("Debug: Converting y_true to pandas Series")
            y_true = pd.Series(y_true)
        if not isinstance(y_pred_series, pd.Series):
            print("Debug: Converting y_pred_series to pandas Series")
            y_pred_series = pd.Series(y_pred_series)

        # Handle different lengths
        min_len = min(len(y_true), len(y_pred_series))
        print(f"Debug: min_len: {min_len}")
        y_true = y_true.iloc[:min_len]
        y_pred_series = y_pred_series.iloc[:min_len]

        # Debug prints after slicing
        print(f"Debug: Length of y_true after slicing: {len(y_true)}")
        print(f"Debug: Length of y_pred_series after slicing: {len(y_pred_series)}")

        metrics = {
            'MAE': mean_absolute_error(y_true, y_pred_series),
            'MSE': mean_squared_error(y_true, y_pred_series),
            'RMSE': np.sqrt(mean_squared_error(y_true, y_pred_series)),
            'R2': r2_score(y_true, y_pred_series),
            'MAPE': mean_absolute_percentage_error(y_true, y_pred_series) * 100,
            'Max Error': np.max(np.abs(y_true - y_pred_series)),
            'Mean Error': np.mean(y_true - y_pred_series),
            'Std Error': np.std(y_true - y_pred_series),
            'Median Absolute Error': np.median(np.abs(y_true - y_pred_series))
        }

        self.model_metrics[model_name] = metrics
        print(f"Debug: Metrics for {model_name} calculated successfully")
        return metrics



    def _calculate_all_metrics(self):
        """Calculate metrics for all models."""
        for model_name, y_pred in self.model_predictions.items():
            self._calculate_metrics(y_pred, model_name)

    def add_model_predictions(self, y_pred: pd.Series, model_name: str):
        """
        Add predictions for a specific model.
        
        Args:
            y_pred: Predicted values
            model_name: Name of the model
        """
        self.model_predictions[model_name] = y_pred
        if self.y_true is not None:
            self._calculate_metrics(y_pred, model_name)

    def _get_display_name(self, model_name: str) -> str:
        """Get display name for a model."""
        return self.VARIANT_DISPLAY_NAMES.get(model_name, model_name)

    def _get_model_color(self, model_name: str) -> str:
        """Get color for a specific model."""
        return self.MODEL_COLORS.get(model_name, self.COLORS['neutral'])

    def _save_fig(self, filename: str):
        """Save the current matplotlib figure to viz_path."""
        if self.viz_path is not None:
            save_path = self.viz_path / filename
            plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor='white')
            print(f"📁 Saved plot: {save_path}")

    # =========================================================================
    # BEFORE MODELLING - EDA PLOTS
    # =========================================================================

    def plot_target_distribution(self, bins: int = 50):
        """Plot target variable distribution with statistics."""
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Histogram with KDE
        axes[0].hist(self.df[self.target], bins=bins, color=self.COLORS['secondary'], 
                    alpha=0.7, edgecolor='black')
        axes[0].set_title(f"Distribution of {self.target}", fontsize=14, fontweight='bold')
        axes[0].set_xlabel(self.target, fontsize=12)
        axes[0].set_ylabel("Frequency", fontsize=12)
        axes[0].grid(True, alpha=0.3)
        
        # Add statistics
        mean_val = self.df[self.target].mean()
        median_val = self.df[self.target].median()
        axes[0].axvline(mean_val, color=self.COLORS['accent'], linestyle='--', 
                       linewidth=2, label=f'Mean: {mean_val:.2f}')
        axes[0].axvline(median_val, color=self.COLORS['highlight'], linestyle='--', 
                       linewidth=2, label=f'Median: {median_val:.2f}')
        axes[0].legend()
        
        # Box plot
        box = axes[1].boxplot(self.df[self.target], vert=True, patch_artist=True)
        box['boxes'][0].set_facecolor(self.COLORS['secondary'])
        box['boxes'][0].set_alpha(0.7)
        axes[1].set_title(f"Box Plot of {self.target}", fontsize=14, fontweight='bold')
        axes[1].set_ylabel(self.target, fontsize=12)
        axes[1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        self._save_fig("01_target_distribution.png")
        plt.show()

    def plot_feature_correlation(self, top_n: int = 20, method: str = 'pearson'):
        """
        Plot feature correlation heatmap.
        
        Args:
            top_n: Number of top correlated features to show
            method: Correlation method ('pearson', 'spearman', 'kendall')
        """
        plt.figure(figsize=(14, 12))
        
        # Calculate correlation
        corr = self.df.corr(numeric_only=True, method=method)
        
        # Get top correlations with target
        if self.target in corr.columns:
            target_corr = corr[self.target].abs().sort_values(ascending=False)
            top_features = target_corr.head(top_n).index.tolist()
            
            # Subset correlation matrix
            corr_subset = corr.loc[top_features, top_features]
            
            # Create heatmap
            mask = np.triu(np.ones_like(corr_subset, dtype=bool), k=1)
            sns.heatmap(corr_subset, mask=mask, annot=True, fmt='.2f', 
                       cmap='RdBu_r', center=0, square=True,
                       linewidths=0.5, cbar_kws={"shrink": 0.8})
            
            plt.title(f"Feature Correlation Heatmap (Top {top_n}, {method.capitalize()})", 
                     fontsize=14, fontweight='bold')
            plt.tight_layout()
            self._save_fig("02_feature_correlation_heatmap.png")
            plt.show()
        else:
            print(f"Warning: Target column '{self.target}' not found in correlation matrix")

    def plot_target_time_series(self, n_points: int = 500, rolling_window: int = 7):
        """
        Plot target variable over time with rolling statistics.
        
        Args:
            n_points: Number of points to display
            rolling_window: Window size for rolling statistics
        """
        if "date" not in self.df.columns:
            raise ValueError("❌ 'date' column required for time series plot.")

        df_sorted = self.df.sort_values("date").head(n_points).copy()
        
        # Calculate rolling statistics
        df_sorted['rolling_mean'] = df_sorted[self.target].rolling(window=rolling_window).mean()
        df_sorted['rolling_std'] = df_sorted[self.target].rolling(window=rolling_window).std()
        
        fig, axes = plt.subplots(2, 1, figsize=(16, 10))
        
        # Time series plot
        axes[0].plot(df_sorted["date"], df_sorted[self.target], 
                    color=self.COLORS['primary'], alpha=0.6, linewidth=1, label='Actual')
        axes[0].plot(df_sorted["date"], df_sorted['rolling_mean'], 
                    color=self.COLORS['accent'], linewidth=2, 
                    label=f'{rolling_window}-day Rolling Mean')
        axes[0].fill_between(df_sorted["date"],
                             df_sorted['rolling_mean'] - df_sorted['rolling_std'],
                             df_sorted['rolling_mean'] + df_sorted['rolling_std'],
                             color=self.COLORS['secondary'], alpha=0.2,
                             label=f'{rolling_window}-day Rolling Std')
        axes[0].set_title(f"Time Series of {self.target} with Rolling Statistics", 
                         fontsize=14, fontweight='bold')
        axes[0].set_xlabel("Date", fontsize=12)
        axes[0].set_ylabel(self.target, fontsize=12)
        axes[0].legend(loc='best')
        axes[0].grid(True, alpha=0.3)
        
        # Distribution over time (violin plot by month)
        if len(df_sorted) > 30:
            df_sorted['month'] = df_sorted['date'].dt.to_period('M').astype(str)
            months = df_sorted['month'].unique()[:12]  # Limit to 12 months
            data_by_month = [df_sorted[df_sorted['month'] == m][self.target].values 
                            for m in months]
            
            parts = axes[1].violinplot(data_by_month, positions=range(len(months)),
                                       showmeans=True, showmedians=True)
            
            for pc in parts['bodies']:
                pc.set_facecolor(self.COLORS['secondary'])
                pc.set_alpha(0.7)
            
            axes[1].set_xticks(range(len(months)))
            axes[1].set_xticklabels(months, rotation=45)
            axes[1].set_title(f"Distribution of {self.target} by Month", 
                            fontsize=14, fontweight='bold')
            axes[1].set_ylabel(self.target, fontsize=12)
            axes[1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        self._save_fig("03_target_time_series.png")
        plt.show()

    def plot_seasonal_decomposition(self, period: int = 7):
        """
        Plot seasonal decomposition of the target variable.
        
        Args:
            period: Seasonality period (7 for weekly, 365 for yearly)
        """
        if "date" not in self.df.columns:
            raise ValueError("❌ 'date' column required for seasonal decomposition.")
        
        from statsmodels.tsa.seasonal import seasonal_decompose
        
        df_sorted = self.df.sort_values("date").copy()
        df_sorted = df_sorted.set_index('date')
        
        # Ensure regular frequency
        df_sorted = df_sorted.asfreq('D', method='ffill')
        
        try:
            decomposition = seasonal_decompose(df_sorted[self.target], 
                                              model='additive', period=period)
            
            fig, axes = plt.subplots(4, 1, figsize=(16, 12))
            
            # Original
            axes[0].plot(decomposition.observed, color=self.COLORS['primary'], linewidth=1)
            axes[0].set_title("Original", fontsize=12, fontweight='bold')
            axes[0].set_ylabel(self.target, fontsize=10)
            axes[0].grid(True, alpha=0.3)
            
            # Trend
            axes[1].plot(decomposition.trend, color=self.COLORS['accent'], linewidth=2)
            axes[1].set_title("Trend", fontsize=12, fontweight='bold')
            axes[1].set_ylabel(self.target, fontsize=10)
            axes[1].grid(True, alpha=0.3)
            
            # Seasonal
            axes[2].plot(decomposition.seasonal, color=self.COLORS['secondary'], linewidth=1)
            axes[2].set_title(f"Seasonal (Period={period})", fontsize=12, fontweight='bold')
            axes[2].set_ylabel(self.target, fontsize=10)
            axes[2].grid(True, alpha=0.3)
            
            # Residual
            axes[3].plot(decomposition.resid, color=self.COLORS['highlight'], linewidth=1)
            axes[3].set_title("Residual", fontsize=12, fontweight='bold')
            axes[3].set_ylabel(self.target, fontsize=10)
            axes[3].set_xlabel("Date", fontsize=12)
            axes[3].grid(True, alpha=0.3)
            
            plt.suptitle(f"Seasonal Decomposition of {self.target}", 
                        fontsize=14, fontweight='bold')
            plt.tight_layout()
            self._save_fig("04_seasonal_decomposition.png")
            plt.show()
            
        except Exception as e:
            warnings.warn(f"Seasonal decomposition failed: {e}")

    # =========================================================================
    # AFTER MODELLING - EVALUATION PLOTS (SINGLE MODEL)
    # =========================================================================

    def plot_actual_vs_predicted(self, model_name: str = "linear_regression", show_density: bool = True):
        """
        Plot actual vs predicted values with perfect prediction line for a specific model.
        
        Args:
            model_name: Name of the model to plot
            show_density: Whether to show density contours
        """
        if self.y_true is None:
            raise ValueError("❌ y_true required for after-modelling plots.")
        
        if model_name not in self.model_predictions:
            raise ValueError(f"❌ Model '{model_name}' not found in predictions.")
        
        y_pred = self.model_predictions[model_name]
        display_name = self._get_display_name(model_name)

        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        
        # Scatter plot
        if show_density:
            # Hexbin for density
            hb = axes[0].hexbin(self.y_true, y_pred, gridsize=50, 
                               cmap='Blues', alpha=0.6, mincnt=1)
            plt.colorbar(hb, ax=axes[0], label='Count')
        else:
            axes[0].scatter(self.y_true, y_pred, alpha=0.5, 
                          color=self._get_model_color(model_name), s=20)
        
        # Perfect prediction line
        min_val = min(self.y_true.min(), y_pred.min())
        max_val = max(self.y_true.max(), y_pred.max())
        axes[0].plot([min_val, max_val], [min_val, max_val], 
                    'r--', lw=2, label="Perfect Prediction")
        
        # Calculate R²
        r2 = r2_score(self.y_true, y_pred)
        
        axes[0].set_title(f"Actual vs Predicted - {display_name} (R² = {r2:.4f})", 
                         fontsize=14, fontweight='bold')
        axes[0].set_xlabel("Actual", fontsize=12)
        axes[0].set_ylabel("Predicted", fontsize=12)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Error distribution
        errors = self.y_true - y_pred
        axes[1].hist(errors, bins=50, color=self._get_model_color(model_name), 
                    alpha=0.7, edgecolor='black')
        axes[1].axvline(0, color='red', linestyle='--', linewidth=2, label='Zero Error')
        axes[1].axvline(errors.mean(), color='green', linestyle='--', linewidth=2, 
                       label=f'Mean: {errors.mean():.2f}')
        axes[1].set_title(f"Prediction Error Distribution - {display_name}", 
                         fontsize=14, fontweight='bold')
        axes[1].set_xlabel("Error (Actual - Predicted)", fontsize=12)
        axes[1].set_ylabel("Frequency", fontsize=12)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        self._save_fig(f"05_actual_vs_predicted_{model_name}.png")
        plt.show()

    def plot_residuals_analysis(self, model_name: str = "linear_regression"):
        """
        Comprehensive residual analysis plots for a specific model.
        
        Args:
            model_name: Name of the model to analyze
        """
        if self.y_true is None:
            raise ValueError("❌ y_true required for residual analysis.")
        
        if model_name not in self.model_predictions:
            raise ValueError(f"❌ Model '{model_name}' not found in predictions.")
        
        y_pred = self.model_predictions[model_name]
        display_name = self._get_display_name(model_name)
        residuals = self.y_true - y_pred
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Residuals vs Predicted
        axes[0, 0].scatter(y_pred, residuals, alpha=0.5, 
                          color=self._get_model_color(model_name), s=20)
        axes[0, 0].axhline(0, color='red', linestyle='--', linewidth=2)
        axes[0, 0].set_title(f"Residuals vs Predicted - {display_name}", 
                            fontsize=12, fontweight='bold')
        axes[0, 0].set_xlabel("Predicted", fontsize=11)
        axes[0, 0].set_ylabel("Residuals", fontsize=11)
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Residuals Distribution with Normal curve
        axes[0, 1].hist(residuals, bins=50, density=True, 
                       color=self._get_model_color(model_name), alpha=0.7, edgecolor='black')
        
        # Fit normal distribution
        mu, sigma = residuals.mean(), residuals.std()
        x = np.linspace(residuals.min(), residuals.max(), 100)
        axes[0, 1].plot(x, stats.norm.pdf(x, mu, sigma), 
                       'r-', linewidth=2, label=f'Normal(μ={mu:.2f}, σ={sigma:.2f})')
        axes[0, 1].set_title(f"Residuals Distribution - {display_name}", 
                            fontsize=12, fontweight='bold')
        axes[0, 1].set_xlabel("Residuals", fontsize=11)
        axes[0, 1].set_ylabel("Density", fontsize=11)
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Q-Q Plot
        stats.probplot(residuals, dist="norm", plot=axes[1, 0])
        axes[1, 0].get_lines()[0].set_color(self._get_model_color(model_name))
        axes[1, 0].get_lines()[0].set_markersize(4)
        axes[1, 0].get_lines()[1].set_color('red')
        axes[1, 0].get_lines()[1].set_linewidth(2)
        axes[1, 0].set_title(f"Q-Q Plot - {display_name}", 
                            fontsize=12, fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Scale-Location Plot (Standardized residuals)
        standardized_residuals = residuals / residuals.std()
        axes[1, 1].scatter(y_pred, np.abs(standardized_residuals), 
                          alpha=0.5, color=self._get_model_color(model_name), s=20)
        axes[1, 1].set_title(f"Scale-Location Plot - {display_name}", 
                            fontsize=12, fontweight='bold')
        axes[1, 1].set_xlabel("Predicted", fontsize=11)
        axes[1, 1].set_ylabel("|Standardized Residuals|", fontsize=11)
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.suptitle(f"Comprehensive Residual Analysis - {display_name}", 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        self._save_fig(f"06_residuals_analysis_{model_name}.png")
        plt.show()

    # =========================================================================
    # MULTI-MODEL COMPARISON PLOTS
    # =========================================================================

    def plot_multi_model_comparison(self, metrics_to_plot: List[str] = None):
        """
        Compare multiple models across various metrics.
        
        Args:
            metrics_to_plot: List of metrics to plot (default: ['MAE', 'RMSE', 'R2', 'MAPE'])
        """
        if not self.model_metrics:
            raise ValueError("❌ No model metrics available. Run predictions first.")
        
        if metrics_to_plot is None:
            metrics_to_plot = ['MAE', 'RMSE', 'R2', 'MAPE']
        
        # Prepare data for plotting
        models = list(self.model_metrics.keys())
        display_names = [self._get_display_name(m) for m in models]
        colors = [self._get_model_color(m) for m in models]
        
        # Create subplots
        n_metrics = len(metrics_to_plot)
        fig, axes = plt.subplots(1, n_metrics, figsize=(5*n_metrics, 6))
        
        if n_metrics == 1:
            axes = [axes]
        
        for idx, metric in enumerate(metrics_to_plot):
            # Collect metric values
            metric_values = []
            for model in models:
                if metric in self.model_metrics[model]:
                    metric_values.append(self.model_metrics[model][metric])
                else:
                    metric_values.append(np.nan)
            
            # Create bar chart
            bars = axes[idx].bar(display_names, metric_values, color=colors, alpha=0.7)
            axes[idx].set_title(f"{metric} Comparison", fontsize=12, fontweight='bold')
            axes[idx].set_ylabel(metric, fontsize=11)
            axes[idx].tick_params(axis='x', rotation=45)
            axes[idx].grid(True, alpha=0.3, axis='y')
            
            # Add value labels on bars
            for bar, value in zip(bars, metric_values):
                height = bar.get_height()
                axes[idx].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                              f'{value:.4f}', ha='center', va='bottom', fontsize=9)
        
        plt.suptitle("Model Performance Comparison", fontsize=14, fontweight='bold')
        plt.tight_layout()
        self._save_fig("07_multi_model_comparison.png")
        plt.show()

    def plot_multi_model_actual_vs_predicted(self, n_points: int = 200):
        """
        Plot actual vs predicted for all models in a grid.
        Handles numpy arrays and pandas Series safely.
        """
        if self.y_true is None:
            raise ValueError("❌ y_true required.")
        
        if not self.model_predictions:
            raise ValueError("❌ No model predictions available.")
        
        # Ensure y_true is a Series
        y_true = self.y_true
        if not isinstance(y_true, pd.Series):
            y_true = pd.Series(y_true)

        models = list(self.model_predictions.keys())
        n_models = len(models)
        
        # Grid layout
        n_cols = min(2, n_models)
        n_rows = (n_models + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 5*n_rows))
        if n_models == 1:
            axes = np.array([axes])
        axes = axes.flatten()
        
        for idx, model_name in enumerate(models):
            y_pred = self.model_predictions[model_name]

            # Convert numpy arrays to Series
            if not isinstance(y_pred, pd.Series):
                y_pred = pd.Series(y_pred)

            # Align lengths
            min_len = min(len(y_true), len(y_pred))
            y_true_aligned = y_true.iloc[:min_len]
            y_pred_aligned = y_pred.iloc[:min_len]

            # Limit points for clarity
            if min_len > n_points:
                indices = np.random.choice(min_len, n_points, replace=False)
                y_true_subset = y_true_aligned.iloc[indices]
                y_pred_subset = y_pred_aligned.iloc[indices]
            else:
                y_true_subset = y_true_aligned
                y_pred_subset = y_pred_aligned
            
            display_name = self._get_display_name(model_name)
            color = self._get_model_color(model_name)
            
            # Scatter plot
            axes[idx].scatter(y_true_subset, y_pred_subset, alpha=0.5, 
                            color=color, s=20)
            
            # Perfect prediction line
            min_val = min(y_true_subset.min(), y_pred_subset.min())
            max_val = max(y_true_subset.max(), y_pred_subset.max())
            axes[idx].plot([min_val, max_val], [min_val, max_val], 
                        'r--', lw=2, label="Perfect")
            
            # R²
            r2 = r2_score(y_true_subset, y_pred_subset)
            
            axes[idx].set_title(f"{display_name} (R² = {r2:.4f})", 
                                fontsize=12, fontweight='bold')
            axes[idx].set_xlabel("Actual", fontsize=10)
            axes[idx].set_ylabel("Predicted", fontsize=10)
            axes[idx].legend(fontsize=9)
            axes[idx].grid(True, alpha=0.3)
        
        # Hide unused subplots
        for idx in range(n_models, len(axes)):
            axes[idx].axis('off')
        
        plt.suptitle("Actual vs Predicted - All Models", fontsize=14, fontweight='bold')
        plt.tight_layout()
        self._save_fig("08_multi_model_actual_vs_predicted.png")
        plt.show()

    def plot_multi_model_time_series_comparison(self, n_points: int = 100):
        """
        Plot time series comparison for all models.
        
        Args:
            n_points: Number of points to display
        """
        if self.y_true is None or self.dates is None:
            raise ValueError("❌ y_true and dates required.")
        
        if not self.model_predictions:
            raise ValueError("❌ No model predictions available.")
        
        # Prepare data
        df = pd.DataFrame({
            'date': self.dates,
            'actual': self.y_true
        })
        
        # Add predictions for each model
        for model_name, y_pred in self.model_predictions.items():
            df[self._get_display_name(model_name)] = y_pred
        
        # Sort and limit points
        df = df.sort_values('date').head(n_points)
        
        # Create plot
        fig, axes = plt.subplots(2, 1, figsize=(16, 10))
        
        # Time series plot
        axes[0].plot(df['date'], df['actual'], label='Actual', 
                    color=self.COLORS['primary'], linewidth=3, alpha=0.8)
        
        # Plot each model
        models = list(self.model_predictions.keys())
        for model_name in models:
            display_name = self._get_display_name(model_name)
            color = self._get_model_color(model_name)
            axes[0].plot(df['date'], df[display_name], label=display_name,
                        color=color, linewidth=2, alpha=0.7)
        
        axes[0].set_title("Time Series Comparison - All Models", 
                         fontsize=14, fontweight='bold')
        axes[0].set_xlabel("Date", fontsize=12)
        axes[0].set_ylabel(self.target, fontsize=12)
        axes[0].legend(loc='best', fontsize=10)
        axes[0].grid(True, alpha=0.3)
        
        # Error plot
        for model_name in models:
            display_name = self._get_display_name(model_name)
            color = self._get_model_color(model_name)
            error = df['actual'] - df[display_name]
            axes[1].plot(df['date'], error, label=display_name,
                        color=color, linewidth=1.5, alpha=0.7)
        
        axes[1].axhline(0, color='red', linestyle='--', linewidth=2)
        axes[1].set_title("Prediction Errors Over Time", 
                         fontsize=14, fontweight='bold')
        axes[1].set_xlabel("Date", fontsize=12)
        axes[1].set_ylabel("Error (Actual - Predicted)", fontsize=12)
        axes[1].legend(loc='best', fontsize=10)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        self._save_fig("09_multi_model_time_series_comparison.png")
        plt.show()

    def plot_model_performance_radar(self):
        """
        Create a radar chart comparing model performance across multiple metrics.
        """
        if not self.model_metrics:
            raise ValueError("❌ No model metrics available.")
        
        # Select metrics for radar chart
        radar_metrics = ['R2', 'MAE', 'RMSE', 'MAPE', 'Median Absolute Error']
        
        # Normalize metrics (higher is better for R2, lower is better for others)
        normalized_data = {}
        for model_name, metrics in self.model_metrics.items():
            normalized = {}
            for metric in radar_metrics:
                if metric in metrics:
                    if metric == 'R2':
                        # R2: higher is better, normalize to 0-1
                        normalized[metric] = max(0, metrics[metric])
                    else:
                        # Other metrics: lower is better, invert and normalize
                        # Use min-max normalization
                        all_values = [m.get(metric, 0) for m in self.model_metrics.values()]
                        max_val = max(all_values) if all_values else 1
                        min_val = min(all_values) if all_values else 0
                        if max_val > min_val:
                            normalized[metric] = 1 - (metrics[metric] - min_val) / (max_val - min_val)
                        else:
                            normalized[metric] = 1
                else:
                    normalized[metric] = 0
            normalized_data[self._get_display_name(model_name)] = normalized
        
        # Create radar chart
        angles = np.linspace(0, 2 * np.pi, len(radar_metrics), endpoint=False).tolist()
        angles += angles[:1]  # Close the polygon
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        # Plot each model
        for idx, (model_display, data) in enumerate(normalized_data.items()):
            values = [data[metric] for metric in radar_metrics]
            values += values[:1]  # Close the polygon
            color = self._get_model_color(list(self.model_metrics.keys())[idx])
            
            ax.plot(angles, values, 'o-', linewidth=2, label=model_display, color=color)
            ax.fill(angles, values, alpha=0.1, color=color)
        
        # Set labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(radar_metrics, fontsize=11)
        ax.set_ylim(0, 1)
        ax.set_title("Model Performance Radar Chart\n(Higher is better for all metrics)", 
                    fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        ax.grid(True)
        
        plt.tight_layout()
        self._save_fig("10_model_performance_radar.png")
        plt.show()

    def plot_error_distribution_comparison(self, bins: int = 50):
        """
        Compare error distributions across all models.
        
        Args:
            bins: Number of bins for histogram
        """
        if self.y_true is None:
            raise ValueError("❌ y_true required.")
        
        if not self.model_predictions:
            raise ValueError("❌ No model predictions available.")
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Histogram of errors
        for model_name, y_pred in self.model_predictions.items():
            display_name = self._get_display_name(model_name)
            color = self._get_model_color(model_name)
            errors = self.y_true - y_pred
            
            axes[0].hist(errors, bins=bins, alpha=0.5, 
                        label=display_name, color=color, density=True)
        
        axes[0].axvline(0, color='red', linestyle='--', linewidth=2, label='Zero Error')
        axes[0].set_title("Error Distribution Comparison", fontsize=14, fontweight='bold')
        axes[0].set_xlabel("Error (Actual - Predicted)", fontsize=12)
        axes[0].set_ylabel("Density", fontsize=12)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Box plot of errors
        error_data = []
        labels = []
        colors_box = []
        
        for model_name, y_pred in self.model_predictions.items():
            display_name = self._get_display_name(model_name)
            errors = self.y_true - y_pred
            error_data.append(errors)
            labels.append(display_name)
            colors_box.append(self._get_model_color(model_name))
        
        box = axes[1].boxplot(error_data, labels=labels, patch_artist=True)
        for patch, color in zip(box['boxes'], colors_box):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        axes[1].axhline(0, color='red', linestyle='--', linewidth=2)
        axes[1].set_title("Error Statistics Comparison", fontsize=14, fontweight='bold')
        axes[1].set_ylabel("Error (Actual - Predicted)", fontsize=12)
        axes[1].grid(True, alpha=0.3, axis='y')
        axes[1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        self._save_fig("11_error_distribution_comparison.png")
        plt.show()
    
    def plot_all_predictions_overlay_daily(self, n_days: int = 120):
        """
        Creates a subplot grid:
        - One subplot per model: Actual vs that model (daily, aggregated, first N days)
        - Final subplot: Actual vs ALL models overlay
        """

        print("\n[DEBUG][plot_all_predictions_overlay_daily] Starting daily overlay subplot plot...")

        # --- Validation ---
        if self.y_true is None or self.dates is None or not self.model_predictions:
            raise ValueError("❌ y_true, dates, and model predictions are required.")
        print("[DEBUG] y_true, dates, and model_predictions validated.")

        # --- Ensure y_true is a Series ---
        y_true = self.y_true
        if not isinstance(y_true, pd.Series):
            print("[DEBUG] Converting y_true to pandas Series.")
            y_true = pd.Series(y_true)

        # --- Build base DataFrame ---
        df = pd.DataFrame({
            "date": pd.to_datetime(self.dates),
            "actual": y_true.values
        })

        # --- Add predictions ---
        for model_name, y_pred in self.model_predictions.items():
            display_name = self._get_display_name(model_name)

            if not isinstance(y_pred, pd.Series):
                y_pred = pd.Series(y_pred)

            min_len = min(len(df), len(y_pred))
            df[display_name] = y_pred.iloc[:min_len].values
            df[display_name] = df[display_name].fillna(0)

        # --- Aggregate by date ---
        agg_cols = ["actual"] + [self._get_display_name(m) for m in self.model_predictions.keys()]
        daily_df = df.groupby("date")[agg_cols].sum().reset_index()

        # --- Limit to first N days ---
        daily_df = daily_df.sort_values("date").head(n_days)

        # --- Prepare subplot grid ---
        model_names = list(self.model_predictions.keys())
        n_models = len(model_names)

        total_plots = n_models + 1  # +1 for final overlay

        n_cols = 2
        n_rows = (total_plots + 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5 * n_rows))
        axes = axes.flatten()

        # --- 1) Individual model subplots ---
        for idx, model_name in enumerate(model_names):
            ax = axes[idx]
            display_name = self._get_display_name(model_name)
            color = self._get_model_color(model_name)

            ax.plot(
                daily_df["date"],
                daily_df["actual"],
                label="Actual",
                color=self.COLORS["primary"],
                linestyle="--",
                linewidth=2
            )

            ax.plot(
                daily_df["date"],
                daily_df[display_name],
                label=display_name,
                color=color,
                linewidth=1.8
            )

            # UPDATED TITLE
            ax.set_title(
                f"Actual vs {display_name} (Linear Regression) — Daily, First {n_days} Days",
                fontsize=12,
                fontweight="bold"
            )

            ax.set_xlabel("Date")
            ax.set_ylabel(self.target)
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis='x', rotation=45)

        # --- 2) Final subplot: ALL MODELS overlay ---
        ax = axes[n_models]

        ax.plot(
            daily_df["date"],
            daily_df["actual"],
            label="Actual",
            color=self.COLORS["primary"],
            linestyle="--",
            linewidth=2.5
        )

        for model_name in model_names:
            display_name = self._get_display_name(model_name)
            color = self._get_model_color(model_name)

            ax.plot(
                daily_df["date"],
                daily_df[display_name],
                label=display_name,
                color=color,
                linewidth=1.8,
                alpha=0.85
            )

        # UPDATED TITLE
        ax.set_title(
            f"Actual vs ALL Linear Regression Models — Daily, First {n_days} Days",
            fontsize=12,
            fontweight="bold"
        )

        ax.set_xlabel("Date")
        ax.set_ylabel(self.target)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='x', rotation=45)

        # --- Hide unused subplots ---
        for i in range(n_models + 1, len(axes)):
            axes[i].axis("off")

        plt.tight_layout()

        # --- Save ---
        self._save_fig("14_all_predictions_overlay_daily_subplots.png")

        plt.show()
        print("[DEBUG][plot_all_predictions_overlay_daily] Completed.\n")

    def plot_all_predictions_overlay_monthly(self):
        """
        Professional overlay plot of actual values and all model predictions aggregated monthly.
        Handles both pandas Series and numpy arrays gracefully.
        Includes debug print statements for tracing internal steps.
        """
        print("\n[DEBUG][plot_all_predictions_overlay_monthly] Starting monthly overlay plot...")

        # --- Validation ---
        if self.y_true is None or self.dates is None or not self.model_predictions:
            raise ValueError("❌ y_true, dates, and model predictions are required.")
        print("[DEBUG] y_true, dates, and model_predictions validated.")

        # --- Ensure y_true is a Series ---
        y_true = self.y_true
        if not isinstance(y_true, pd.Series):
            print("[DEBUG] Converting y_true to pandas Series.")
            y_true = pd.Series(y_true)
        print(f"[DEBUG] y_true length: {len(y_true)}")

        # --- Build base DataFrame ---
        df = pd.DataFrame({
            "date": pd.to_datetime(self.dates),
            "actual": y_true.values
        })
        print(f"[DEBUG] Base DataFrame shape: {df.shape}")

        # --- Add predictions ---
        for model_name, y_pred in self.model_predictions.items():
            display_name = self._get_display_name(model_name)
            print(f"[DEBUG] Processing model: {model_name} → display name: {display_name}")

            if not isinstance(y_pred, pd.Series):
                print(f"[DEBUG] Converting predictions of {model_name} to pandas Series.")
                y_pred = pd.Series(y_pred)

            min_len = min(len(df), len(y_pred))
            print(f"[DEBUG] Aligning lengths → min_len = {min_len}")

            df[display_name] = y_pred.iloc[:min_len].values
            df[display_name] = df[display_name].fillna(0)  # Ensure inclusion in aggregation
            print(f"[DEBUG] Added column '{display_name}' to DataFrame.")

        # --- Convert to monthly frequency ---
        df["month"] = df["date"].dt.to_period("M").dt.to_timestamp()
        print("[DEBUG] Converted 'date' to 'month' column.")

        # --- Aggregate monthly sums ---
        monthly_df = df.groupby("month").sum(numeric_only=True)
        print(f"[DEBUG] Monthly aggregated DataFrame shape: {monthly_df.shape}")
        print(f"[DEBUG] Columns in monthly_df: {monthly_df.columns.tolist()}")

        # --- Plot ---
        plt.figure(figsize=(18, 9))

        # Actual
        plt.plot(
            monthly_df.index,
            monthly_df["actual"],
            label="Actual (Monthly)",
            color=self.COLORS["primary"],
            linestyle="--",
            linewidth=2.5,
            alpha=0.9
        )

        # Predictions
        for model_name in self.model_predictions.keys():
            display_name = self._get_display_name(model_name)
            color = self._get_model_color(model_name)

            if display_name in monthly_df.columns:
                print(f"[DEBUG] Plotting model: {display_name}")
                plt.plot(
                    monthly_df.index,
                    monthly_df[display_name],
                    label=f"{display_name} (Monthly)",
                    color=color,
                    linewidth=1.8,
                    alpha=0.85
                )
            else:
                print(f"[DEBUG] ❌ Column '{display_name}' not found in monthly_df — skipping.")

        # --- Title & labels ---
        plt.title(
            "Monthly Actual vs Predicted Values (All Models)",
            fontsize=16,
            fontweight="bold"
        )
        plt.xlabel("Month", fontsize=13)
        plt.ylabel(self.target, fontsize=13)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)

        # --- Save & show ---
        self._save_fig("14_all_predictions_overlay_monthly.png")
        plt.show()

        print("[DEBUG][plot_all_predictions_overlay_monthly] Completed.\n")


    def plot_cumulative_error_comparison(self):
        """
        Plot cumulative error over time for all models.
        """
        if self.y_true is None or self.dates is None:
            raise ValueError("❌ y_true and dates required.")
        
        if not self.model_predictions:
            raise ValueError("❌ No model predictions available.")
        
        # Prepare data
        df = pd.DataFrame({'date': self.dates, 'actual': self.y_true})
        df = df.sort_values('date')
        
        # Calculate cumulative errors for each model
        for model_name, y_pred in self.model_predictions.items():
            display_name = self._get_display_name(model_name)
            df[f'error_{display_name}'] = df['actual'] - y_pred
            df[f'cum_error_{display_name}'] = df[f'error_{display_name}'].cumsum()
        
        fig, axes = plt.subplots(2, 1, figsize=(16, 10))
        
        # Cumulative error plot
        for model_name in self.model_predictions.keys():
            display_name = self._get_display_name(model_name)
            color = self._get_model_color(model_name)
            axes[0].plot(df['date'], df[f'cum_error_{display_name}'], 
                        label=display_name, color=color, linewidth=2)
        
        axes[0].axhline(0, color='red', linestyle='--', linewidth=2)
        axes[0].set_title("Cumulative Prediction Error Comparison", 
                         fontsize=14, fontweight='bold')
        axes[0].set_xlabel("Date", fontsize=12)
        axes[0].set_ylabel("Cumulative Error", fontsize=12)
        axes[0].legend(loc='best', fontsize=10)
        axes[0].grid(True, alpha=0.3)
        
        # Rolling MAE comparison
        rolling_window = min(30, len(df) // 10)
        
        for model_name in self.model_predictions.keys():
            display_name = self._get_display_name(model_name)
            color = self._get_model_color(model_name)
            rolling_mae = df[f'error_{display_name}'].abs().rolling(window=rolling_window).mean()
            axes[1].plot(df['date'], rolling_mae, 
                        label=f'{display_name} ({rolling_window}-day MAE)', 
                        color=color, linewidth=2, alpha=0.8)
        
        axes[1].set_title(f"Rolling {rolling_window}-day MAE Comparison", 
                         fontsize=14, fontweight='bold')
        axes[1].set_xlabel("Date", fontsize=12)
        axes[1].set_ylabel("MAE", fontsize=12)
        axes[1].legend(loc='best', fontsize=10)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        self._save_fig("12_cumulative_error_comparison.png")
        plt.show()

    def plot_model_improvement_heatmap(self):
        """
        Create a heatmap showing percentage improvement of each model over baseline.
        """
        if not self.model_metrics or "linear_regression" not in self.model_metrics:
            raise ValueError("❌ Baseline model metrics required.")
        
        baseline_metrics = self.model_metrics["linear_regression"]
        models_to_compare = [m for m in self.model_predictions.keys() if m != "linear_regression"]
        
        if not models_to_compare:
            print("⚠️ No models to compare with baseline")
            return
        
        # Metrics to compare
        comparison_metrics = ['MAE', 'RMSE', 'MAPE', 'R2']
        
        # Calculate improvements
        improvement_data = []
        row_labels = []
        
        for model_name in models_to_compare:
            if model_name in self.model_metrics:
                model_metrics = self.model_metrics[model_name]
                improvements = []
                row_labels.append(self._get_display_name(model_name))
                
                for metric in comparison_metrics:
                    if metric in baseline_metrics and metric in model_metrics:
                        baseline_val = baseline_metrics[metric]
                        model_val = model_metrics[metric]
                        
                        if metric == 'R2':
                            # For R2, higher is better
                            improvement = ((model_val - baseline_val) / abs(baseline_val)) * 100
                        else:
                            # For error metrics, lower is better
                            improvement = ((baseline_val - model_val) / baseline_val) * 100
                        improvements.append(improvement)
                    else:
                        improvements.append(np.nan)
                
                improvement_data.append(improvements)
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(12, 6))
        
        im = ax.imshow(improvement_data, cmap='RdYlGn', aspect='auto')
        
        # Add text annotations
        for i in range(len(improvement_data)):
            for j in range(len(comparison_metrics)):
                value = improvement_data[i][j]
                if not np.isnan(value):
                    text = ax.text(j, i, f'{value:.1f}%',  # noqa: F841
                                  ha="center", va="center", color="black" if abs(value) < 50 else "white")
        
        # Set labels
        ax.set_xticks(range(len(comparison_metrics)))
        ax.set_xticklabels(comparison_metrics)
        ax.set_yticks(range(len(row_labels)))
        ax.set_yticklabels(row_labels)
        
        ax.set_title("Percentage Improvement Over Baseline Model\n(Green = Better, Red = Worse)", 
                    fontsize=14, fontweight='bold')
        plt.colorbar(im, ax=ax, label='Improvement (%)')
        
        plt.tight_layout()
        self._save_fig("13_model_improvement_heatmap.png")
        plt.show()

    # =========================================================================
    # BATCH PLOT GENERATION
    # =========================================================================

    def generate_all_eda_plots(self):
        """Generate all EDA (before modelling) plots at once."""
        print("\n" + "=" * 80)
        print("GENERATING ALL EDA PLOTS")
        print("=" * 80 + "\n")
        
        try:
            print("📊 Generating target distribution...")
            self.plot_target_distribution()
        except Exception as e:
            print(f"⚠️ Error in target distribution: {e}")
        
        try:
            print("📊 Generating feature correlation...")
            self.plot_feature_correlation()
        except Exception as e:
            print(f"⚠️ Error in feature correlation: {e}")
        
        try:
            print("📊 Generating time series plot...")
            self.plot_target_time_series()
        except Exception as e:
            print(f"⚠️ Error in time series plot: {e}")
        
        try:
            print("📊 Generating seasonal decomposition...")
            self.plot_seasonal_decomposition()
        except Exception as e:
            print(f"⚠️ Error in seasonal decomposition: {e}")
        
        print("\n✅ EDA plots generation complete!\n")

    def generate_all_single_model_plots(self, model_name: str = "linear_regression"):
        """
        Generate all evaluation plots for a single model.
        
        Args:
            model_name: Name of the model to generate plots for
        """
        if model_name not in self.model_predictions:
            print(f"⚠️ Model '{model_name}' not found in predictions.")
            return
        
        print(f"\n" + "=" * 80)  # noqa: F541
        print(f"GENERATING ALL PLOTS FOR {model_name.upper()}")
        print("=" * 80 + "\n")
        
        try:
            print("📊 Generating actual vs predicted...")
            self.plot_actual_vs_predicted(model_name=model_name)
        except Exception as e:
            print(f"⚠️ Error in actual vs predicted: {e}")
        
        try:
            print("📊 Generating residuals analysis...")
            self.plot_residuals_analysis(model_name=model_name)
        except Exception as e:
            print(f"⚠️ Error in residuals analysis: {e}")

    def generate_all_comparison_plots(self):
        """Generate all multi-model comparison plots at once."""
        if len(self.model_predictions) < 2:
            print("⚠️ Need at least 2 models for comparison plots")
            return
        
        print("\n" + "=" * 80)
        print("GENERATING ALL COMPARISON PLOTS")
        print("=" * 80 + "\n")
        
        try:
            print("📊 Generating multi-model comparison...")
            self.plot_multi_model_comparison()
        except Exception as e:
            print(f"⚠️ Error in multi-model comparison: {e}")
        
        try:
            print("📊 Generating multi-model actual vs predicted...")
            self.plot_multi_model_actual_vs_predicted()
        except Exception as e:
            print(f"⚠️ Error in multi-model actual vs predicted: {e}")
        
        try:
            print("📊 Generating multi-model time series comparison...")
            self.plot_multi_model_time_series_comparison()
        except Exception as e:
            print(f"⚠️ Error in multi-model time series comparison: {e}")
        
        try:
            print("📊 Generating model performance radar...")
            self.plot_model_performance_radar()
        except Exception as e:
            print(f"⚠️ Error in model performance radar: {e}")
        
        try:
            print("📊 Generating error distribution comparison...")
            self.plot_error_distribution_comparison()
        except Exception as e:
            print(f"⚠️ Error in error distribution comparison: {e}")
        
        try:
            print("📊 Generating cumulative error comparison...")
            self.plot_cumulative_error_comparison()
        except Exception as e:
            print(f"⚠️ Error in cumulative error comparison: {e}")
        
        try:
            print("📊 Generating model improvement heatmap...")
            self.plot_model_improvement_heatmap()
        except Exception as e:
            print(f"⚠️ Error in model improvement heatmap: {e}")
        
        print("\n✅ Comparison plots generation complete!\n")

    def generate_comprehensive_report(self):
        """Generate a comprehensive report with all plots."""
        print("\n" + "=" * 80)
        print("GENERATING COMPREHENSIVE VISUALIZATION REPORT")
        print("=" * 80 + "\n")
        
        # Generate EDA plots
        self.generate_all_eda_plots()
        
        # Generate single model plots for each model
        for model_name in self.model_predictions.keys():
            self.generate_all_single_model_plots(model_name)
        
        # Generate comparison plots if multiple models
        if len(self.model_predictions) > 1:
            self.generate_all_comparison_plots()
        
        print("\n" + "=" * 80)
        print("REPORT GENERATION COMPLETE!")
        print("=" * 80 + "\n")

    # =========================================================================
    # UTILITY METHODS
    # =========================================================================

    def get_model_metrics_table(self) -> pd.DataFrame:
        """Get a DataFrame with metrics for all models."""
        if not self.model_metrics:
            return pd.DataFrame()
        
        metrics_df = pd.DataFrame(self.model_metrics).T
        metrics_df.index = [self._get_display_name(idx) for idx in metrics_df.index]
        return metrics_df.round(4)

    def get_best_model(self, metric: str = 'RMSE', higher_is_better: bool = False) -> str:
        """
        Get the best model based on a specific metric.
        
        Args:
            metric: Metric to use for comparison
            higher_is_better: Whether higher values are better for this metric
            
        Returns:
            Name of the best model
        """
        if not self.model_metrics:
            return None
        
        best_value = None
        best_model = None
        
        for model_name, metrics in self.model_metrics.items():
            if metric in metrics:
                value = metrics[metric]
                
                if best_value is None:
                    best_value = value
                    best_model = model_name
                elif higher_is_better and value > best_value:
                    best_value = value
                    best_model = model_name
                elif not higher_is_better and value < best_value:
                    best_value = value
                    best_model = model_name
        
        return best_model

    def print_model_summary(self):
        """Print a summary of all model performances."""
        if not self.model_metrics:
            print("❌ No model metrics available.")
            return
        
        print("\n" + "=" * 80)
        print("MODEL PERFORMANCE SUMMARY")
        print("=" * 80)
        
        metrics_df = self.get_model_metrics_table()
        if not metrics_df.empty:
            print("\nMetrics Table:")
            print("-" * 80)
            print(metrics_df)
            
            # Find best models for key metrics
            print("\nBest Models:")
            print("-" * 80)
            for metric, higher_better in [('R2', True), ('MAE', False), 
                                         ('RMSE', False), ('MAPE', False)]:
                if metric in metrics_df.columns:
                    best_model = self.get_best_model(metric, higher_better)
                    if best_model:
                        best_value = self.model_metrics[best_model][metric]
                        print(f"{metric}: {self._get_display_name(best_model)} = {best_value:.4f}")
        
        print("\n" + "=" * 80)