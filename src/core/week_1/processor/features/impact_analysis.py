# src/core/week_1/impact_analysis.py
# src/core/week_1/impact_analysis.py

import os
from typing import Dict, Any
import pandas as pd  # type: ignore
import numpy as np  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
from IPython.display import display  # type: ignore
from statsmodels.tsa.stattools import grangercausalitytests  # type: ignore
from scipy.stats import zscore  # type: ignore

from src.utils import get_path


class ImpactAnalysis:
    """
    ImpactAnalysis for external drivers (holidays, oil price, promotions).
    - merge helpers
    - lagged cross-correlation summary
    - simple change-point detection (rolling z-score on mean diff)
    - Granger causality wrapper
    - plot helpers
    """

    def __init__(self, df_sales: pd.DataFrame, week: int = 1):
        self.df_sales = df_sales.copy()
        self.week = week
        self.eda_path = get_path("eda", week=self.week)
        self.processed_path = get_path("features", week=self.week)
        self.results_stats_path = get_path("features_results", week=self.week)
        self.viz_path = get_path("features_viz", week=self.week)
        os.makedirs(self.eda_path, exist_ok=True)
        os.makedirs(self.processed_path, exist_ok=True)
        os.makedirs(self.results_stats_path, exist_ok=True)
        os.makedirs(self.viz_path, exist_ok=True)
        print(f"[DEBUG] Initialized ImpactAnalysis with sales DataFrame shape {self.df_sales.shape}")

    # ---------------------------
    # Merge Methods
    # ---------------------------
    def merge_holidays(self, df_holidays: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
        df_h = df_holidays.copy()
        df_h[date_col] = pd.to_datetime(df_h[date_col])
        print(f"[DEBUG] Holiday DataFrame shape: {df_h.shape}")
        merged = pd.merge(self.df_sales, df_h[[date_col, "type"]], on=date_col, how="left")
        print(f"[DEBUG] Merged holiday events into sales data. New shape: {merged.shape}")
        return merged

    def merge_perishables(self, df_items: pd.DataFrame) -> pd.DataFrame:
        df_i = df_items.copy()
        print(f"[DEBUG] Items DataFrame shape: {df_i.shape}")
        merged = pd.merge(self.df_sales, df_i[["item_nbr", "perishable"]], on="item_nbr", how="left")
        
        # Check for successful merge/missing values before casting
        missing_perish = merged["perishable"].isna().sum()
        if missing_perish > 0:
            print(f"[WARN] {missing_perish} items had missing 'perishable' flag after merge.")
            
        merged["perishable"] = merged["perishable"].astype(bool)
        print(f"[DEBUG] Merged perishable flag into sales data. New shape: {merged.shape}")
        return merged

    def merge_oil_prices(self, df_oil: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
        df_oil = df_oil.copy()
        df_oil[date_col] = pd.to_datetime(df_oil[date_col])
        print(f"[DEBUG] Oil DataFrame shape: {df_oil.shape}")
        
        # Aggregate sales before merge to match daily oil price
        sales_by_date = self.df_sales.groupby(date_col)["unit_sales"].sum().reset_index()
        print(f"[DEBUG] Aggregated sales data to daily level. Shape: {sales_by_date.shape}")
        
        merged = pd.merge(sales_by_date, df_oil[[date_col, "dcoilwtico"]], on=date_col, how="left")
        
        missing_oil = merged["dcoilwtico"].isna().sum()
        print(f"[DEBUG] Merged oil prices into aggregated sales data. New shape: {merged.shape}")
        print(f"[INFO] Merged data contains {missing_oil} missing oil price values (expected during holidays/weekends).")
        return merged

    # ---------------------------
    # Change-point detection (simple rolling z-score)
    # ---------------------------
    def detect_change_points(self, ts: pd.Series, window: int = 14, z_thresh: float = 3.0) -> pd.DataFrame:
        """
        A lightweight change-point detector:
        - computes rolling mean diff, then zscore, flags points where |z| > z_thresh
        Returns DataFrame with date and zscore and flag.
        """
        print(f"[DEBUG] Running change-point detection on '{ts.name}' with window={window}, z_thresh={z_thresh}")
        s = ts.copy().dropna()
        print(f"[DEBUG] Time Series length (after dropna): {len(s)}")
        
        rolling_mean = s.rolling(window=window, min_periods=1).mean()
        diff = rolling_mean.diff().fillna(0)
        z = zscore(diff.fillna(0))
        res = pd.DataFrame({"date": s.index, "diff": diff.values, "z": z})
        res["is_change"] = np.abs(res["z"]) > z_thresh
        changes = res[res["is_change"]]
        
        print(f"[DEBUG] Detected {len(changes)} candidate change points (z_thresh={z_thresh})")
        display(changes.head(10))
        
        # Save change point stats
        save_path = os.path.join(self.results_stats_path, f"change_points_{ts.name}.csv")
        res.to_csv(save_path, index=False)
        print(f"💾 Change point results saved to {save_path}")
        
        return res

    # ---------------------------
    # Cross-correlation / Lagged impact
    # ---------------------------
    def lagged_cross_correlation(self, x: pd.Series, y: pd.Series, max_lag: int = 30) -> pd.DataFrame:
        """
        Compute cross-correlation function (x leads y by positive lag).
        Returns DataFrame with lag and correlation.
        """
        print(f"[DEBUG] Computing cross-correlation between '{x.name}' (X) and '{y.name}' (Y)")
        print(f"[DEBUG] Max lag: {max_lag} days")
        
        x = x.dropna()
        y = y.dropna()
        
        # align on intersection of indexes (dates)
        idx = x.index.intersection(y.index)
        x, y = x.loc[idx], y.loc[idx]
        print(f"[DEBUG] Aligned series length: {len(x)}")
        
        corrs = []
        for lag in range(-max_lag, max_lag + 1):
            if lag < 0:
                # y leads x
                val = np.corrcoef(x[-lag:].values, y[:len(y)+lag].values)[0, 1]
            elif lag > 0:
                val = np.corrcoef(x[:-lag].values, y[lag:].values)[0, 1]
            else:
                val = np.corrcoef(x.values, y.values)[0, 1]
            corrs.append({"lag": lag, "corr": float(val)})
            
        dfc = pd.DataFrame(corrs).sort_values("lag").reset_index(drop=True)
        display(dfc.head(10))
        
        # save stat
        save_path = os.path.join(self.results_stats_path, f"crosscorr_{x.name}_vs_{y.name}.csv")
        dfc.to_csv(save_path, index=False)
        print(f"💾 Cross-correlation saved to {save_path}")
        
        # Trigger diagram showing the correlation over time 
        
        return dfc

    # ---------------------------
    # Granger causality wrapper
    # ---------------------------
    def run_granger(self, df: pd.DataFrame, target_col: str, exog_col: str, maxlag: int = 14) -> Dict[str, Any]:
        """
        Run Granger causality tests (statsmodels).
        Returns a dict with test summaries for each lag.
        """
        print(f"[DEBUG] Running Granger Causality: Is '{exog_col}' causing '{target_col}'?")
        print(f"[DEBUG] Max lag: {maxlag} periods")
        
        # Prepare input: two-column DataFrame [target, exog]
        tmp = df[[target_col, exog_col]].dropna()
        
        if len(tmp) < maxlag * 2:
            print(f"[WARN] Not enough observations ({len(tmp)}) for Granger causality with maxlag={maxlag}; returning empty result.")
            return {}
            
        data = tmp[[target_col, exog_col]]
        print(f"[DEBUG] Granger test data size: {len(data)}")
        
        try:
            # Note: grangercausalitytests tests the null hypothesis that 'exog_col' DOES NOT Granger-cause 'target_col'.
            res = grangercausalitytests(data, maxlag=maxlag, verbose=False)
        except Exception as e:
            print(f"[ERROR] Granger causality failed: {e}. Check for highly collinear data or lack of variance.")
            return {}

        # Summarize p-values for the F-test at each lag
        summary = {}
        for lag, output in res.items():
            test_res = output[0]['ssr_ftest']  # (ftest_val, pvalue, df_denom, df_num)
            summary[lag] = {"fstat": float(test_res[0]), "pvalue": float(test_res[1])}
            
        summary_df = pd.DataFrame.from_dict(summary, orient="index").rename_axis("lag").reset_index()
        display(summary_df)
        
        # save stat
        save_path = os.path.join(self.results_stats_path, f"granger_{target_col}_vs_{exog_col}.csv")
        summary_df.to_csv(save_path, index=False)
        print(f"💾 Granger causality summary saved to {save_path}")
        
        # Trigger diagram showing the concept of causality 
        
        return summary

    # ---------------------------
    # Plot helpers
    # ---------------------------
    def plot_oil_vs_sales(self, merged_df: pd.DataFrame, date_col: str = "date", sales_col: str = "unit_sales", oil_col: str = "dcoilwtico"):
        merged_df[date_col] = pd.to_datetime(merged_df[date_col])
        print(f"[DEBUG] Plotting '{oil_col}' vs '{sales_col}' (aggregated sales)")
        
        fig, ax1 = plt.subplots(figsize=(12, 6))
        
        ax1.set_xlabel("Date")
        ax1.set_ylabel("Oil Price", color="tab:blue")
        ax1.plot(merged_df[date_col], merged_df[oil_col], color="tab:blue", label="Oil Price")
        
        ax2 = ax1.twinx()
        ax2.set_ylabel("Unit Sales", color="tab:green")
        ax2.plot(merged_df[date_col], merged_df[sales_col], color="tab:green", label="Unit Sales")
        
        plt.title("Oil Price vs Unit Sales Over Time", fontsize=14)
        
        save_path = os.path.join(self.viz_path, "oil_vs_sales.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()
        print(f"💾 Oil vs Sales plot saved to {save_path}")
    
    def plot_holiday_vs_nonholiday_sales_dual(
        self,
        merged_df: pd.DataFrame,
        date_col: str = "date",
        sales_col: str = "unit_sales",
        holiday_col: str = "type",
    ):
        """
        Plot average sales for holidays vs non-holidays using two axes.
        - Left axis: Holiday average sales
        - Right axis: Non-Holiday average sales
        """

        print("[DEBUG] Plotting Holiday vs Non-Holiday sales with dual axes")

        df = merged_df.copy()
        df[date_col] = pd.to_datetime(df[date_col])

        # Binary holiday flag
        df["is_holiday"] = df[holiday_col].notna()

        # Compute average sales
        summary = df.groupby("is_holiday")[sales_col].mean().reset_index()
        summary["day_type"] = summary["is_holiday"].replace({True: "Holiday", False: "Non-Holiday"})

        print("[DEBUG] Summary of average sales:")
        display(summary)

        fig, ax1 = plt.subplots(figsize=(8, 5))

        # Left axis: Holiday average sales
        holiday_avg = summary.loc[summary["is_holiday"] == True, sales_col].values[0]  # noqa: E712
        ax1.bar("Holiday", holiday_avg, color="orange", alpha=0.7)
        ax1.set_ylabel("Holiday Avg Sales", color="orange")
        ax1.tick_params(axis='y', labelcolor="orange")
        ax1.set_ylim(0, max(summary[sales_col])*1.2)
        ax1.set_xlabel("Day Type")
        ax1.set_xticks([0])
        ax1.set_xticklabels(["Holiday vs Non-Holiday"])

        # Right axis: Non-Holiday average sales
        ax2 = ax1.twinx()
        nonholiday_avg = summary.loc[summary["is_holiday"] == False, sales_col].values[0]  # noqa: E712
        ax2.bar("Non-Holiday", nonholiday_avg, color="skyblue", alpha=0.7)
        ax2.set_ylabel("Non-Holiday Avg Sales", color="skyblue")
        ax2.tick_params(axis='y', labelcolor="skyblue")
        ax2.set_ylim(0, max(summary[sales_col])*1.2)

        plt.title("Holiday vs Non-Holiday Average Sales (Dual Axis)")
        plt.tight_layout()

        save_path = os.path.join(self.viz_path, "holiday_vs_nonholiday_dual_axis.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()
        print(f"💾 Holiday vs Non-Holiday dual-axis plot saved to {save_path}")


    # ---------------------------
    # Save merged for reuse
    # ---------------------------
    def save_merged(self, merged_df: pd.DataFrame, filename: str):
        save_path = os.path.join(self.processed_path, filename)
        merged_df.to_csv(save_path, index=False)
        print(f"💾 Merged dataset saved to {save_path}")
        return save_path