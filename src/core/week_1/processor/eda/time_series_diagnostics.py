# src/core/week_1/processor/eda/time_series_diagnostics.py

import os
import pandas as pd  # type: ignore
import matplotlib.pyplot as plt # type: ignore
from pandas.plotting import autocorrelation_plot # type: ignore
from statsmodels.tsa.stattools import adfuller # type: ignore
from statsmodels.tsa.seasonal import STL # type: ignore
import numpy as np # type: ignore
from src.utils import get_path


class TimeSeriesDiagnostics:
    """
    Professional class for advanced time-series diagnostics on Favorita Grocery Sales.

    Responsibilities:
      • Visualize autocorrelation of daily sales.
      • Check stationarity with rolling mean/std and Augmented Dickey-Fuller (ADF) test.
      • Perform STL decomposition to separate trend, seasonality, and residuals.
      • Quantify strength of trend and seasonality.
      • Save diagnostic plots and results to project-standard EDA path.
    """

    def __init__(self, df_sales: pd.DataFrame, date_col="date", sales_col="unit_sales"):
        self.df_sales = df_sales.copy()
        self.date_col = date_col
        self.sales_col = sales_col
        self.eda_path = get_path("eda", week=1)
        os.makedirs(self.eda_path, exist_ok=True)

        # Aggregate daily sales
        self.df_sales[self.date_col] = pd.to_datetime(self.df_sales[self.date_col])
        self.sales_by_date = self.df_sales.groupby(self.date_col)[self.sales_col].sum()
        print(f"[DEBUG] Initialized TimeSeriesDiagnostics with {len(self.sales_by_date)} daily records")

    # ---------------------------
    # Autocorrelation
    # ---------------------------
    def plot_autocorrelation(self):
        plt.figure(figsize=(10, 5))
        autocorrelation_plot(self.sales_by_date)
        plt.title("Autocorrelation of Daily Unit Sales", fontsize=16)
        save_path = os.path.join(self.eda_path, "autocorrelation.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"💾 Autocorrelation plot saved to {save_path}")
        plt.show()

    # ---------------------------
    # Stationarity Checks
    # ---------------------------
    def plot_stationarity_checks(self, window=12):
        rolling_mean = self.sales_by_date.rolling(window=window).mean()
        rolling_std = self.sales_by_date.rolling(window=window).std()

        plt.figure(figsize=(12, 5))
        plt.plot(self.sales_by_date, label="Original")
        plt.plot(rolling_mean, label="Rolling Mean", color="red")
        plt.plot(rolling_std, label="Rolling Std", color="green")
        plt.title("Rolling Mean & Standard Deviation", fontsize=16)
        plt.legend()
        save_path = os.path.join(self.eda_path, "rolling_mean_std.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"💾 Rolling mean/std plot saved to {save_path}")
        plt.show()

    def run_adf_test(self):
        result = adfuller(self.sales_by_date)
        output = {
            "ADF Statistic": result[0],
            "p-value": result[1],
            "lags_used": result[2],
            "n_obs": result[3]
        }
        print("[DEBUG] Augmented Dickey-Fuller Test Results:", output)
        return output

    # ---------------------------
    # STL Decomposition
    # ---------------------------
    def plot_stl_decomposition(self, period=7):
        stl = STL(self.sales_by_date, period=period)
        res = stl.fit()
        res.plot()
        plt.suptitle("STL Decomposition", fontsize=16)
        plt.tight_layout()
        save_path = os.path.join(self.eda_path, "stl_decomposition.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"💾 STL decomposition plot saved to {save_path}")
        plt.show()
        return res

    def calculate_strengths(self, res):
        trend_strength = 1 - (np.var(res.resid) / np.var(res.trend + res.resid))
        seasonal_strength = 1 - (np.var(res.resid) / np.var(res.seasonal + res.resid))
        print(f"[DEBUG] Strength of Trend: {trend_strength:.2f}, Strength of Seasonality: {seasonal_strength:.2f}")
        return {"trend_strength": trend_strength, "seasonal_strength": seasonal_strength}
