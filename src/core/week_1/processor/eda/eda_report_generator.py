#src/core/week_1/processor/eda/eda_report_generator.py


import os

import pandas as pd # type: ignore
import numpy as np # type: ignore
import matplotlib.pyplot as plt # type: ignore
import seaborn as sns # type: ignore
from IPython.display import display # type: ignore

from src.utils import get_path
from .retail_data_cleaner import RetailDataCleaner
from .visualization import Visualization
from .time_series_diagnostics import TimeSeriesDiagnostics


class EDAReportGenerator:
    """
    Optimized EDA workflow for large datasets (34M+ rows).
    
    Key Optimizations:
      • Pre-aggregate data during initialization
      • Minimize repeated groupby operations
      • Use efficient pandas operations
      • Sample data where full granularity isn't needed
    """

    def __init__(self, df: pd.DataFrame, week: int = 1):
        print("[DEBUG] Initializing EDAReportGenerator...")
        self.df = df.copy()
        self.week = week
        self.report = {}
        self.eda_path = get_path("eda", week=week)
        self.stats_path = get_path("eda_stats", week=week)

        os.makedirs(self.eda_path, exist_ok=True)
        os.makedirs(self.stats_path, exist_ok=True)

        # Initialize helper classes
        self.cleaner = RetailDataCleaner()
        
        # OPTIMIZATION: Pass pre_aggregate=True for 10-100x speedup
        print("[OPTIMIZATION] Pre-aggregating data for visualizations...")
        self.visualizer = Visualization(self.df, pre_aggregate=True)
        
        # For diagnostics, we'll aggregate daily first
        print("[OPTIMIZATION] Pre-aggregating data for time series diagnostics...")
        df_daily = self.df.groupby('date')['unit_sales'].sum().reset_index()
        self.diagnostics = TimeSeriesDiagnostics(df_daily)

    # 1. Structural EDA
    def run_structural_summary(self):
        print("➡ Running structural summary...")
        
        # Show dtypes
        self.cleaner.show_dtypes(self.df)
        
        summary = {
            "rows": len(self.df),
            "columns": self.df.shape[1],
            "date_range": (self.df["date"].min(), self.df["date"].max()),
            "unique_stores": self.df["store_nbr"].nunique(),
            "unique_items": self.df["item_nbr"].nunique(),
            "unique_families": self.df["family"].nunique() if "family" in self.df else None,
            "duplicates": self.df.duplicated().sum(),
            "memory_usage_MB": self.df.memory_usage(deep=True).sum() / 1e6
        }
        pd.DataFrame([summary]).to_csv(os.path.join(self.stats_path, "structural_summary.csv"), index=False)
        self.report["structure"] = summary
        display(pd.DataFrame([summary]))
        return summary

    # 2. Data Quality Checks
    def run_data_quality_checks(self):
        print("➡ Running data quality checks...")
        # Build quality report
        quality_report = {
            "numeric_columns": list(self.df.select_dtypes(include=[np.number]).columns),
            "categorical_columns": list(self.df.select_dtypes(include=['object']).columns),
            "date_columns": [col for col in self.df.columns if 'date' in col.lower()],
            "has_negative_sales": (self.df["unit_sales"] < 0).any(),
            "zero_sales_percentage": (self.df["unit_sales"] == 0).mean() * 100,
            "onpromotion_missing_after_fill": self.df["onpromotion"].isna().sum() if "onpromotion" in self.df.columns else None
        }
        pd.DataFrame([quality_report]).to_csv(os.path.join(self.stats_path, "data_quality_report.csv"), index=False)
        self.report["data_quality"] = quality_report
        print("   • Data quality report:")
        display(pd.DataFrame([quality_report]))
        return quality_report
    
    # 3. Missing Value EDA
    def run_missing_value_report(self):
        print("➡ Running missing value report...")
        nulls = self.cleaner.plot_missing_values(self.df)
        self.report["missing_values"] = nulls
        return nulls
    
    # 4. Check for missing calendar days in the dataset and plot coverage.
    def check_and_plot_missing_calendar(self):
        print("➡ Checking and plotting missing calendar days...")
        report_ = self.visualizer.check_and_plot_missing_calendar(self.df, date_col="date")
        self.report["missing_calendar"] = report_
        return report_
    
    # 5. Complete daily coverage
    def complete_daily_coverage(self):
        print("➡ Completing daily coverage by filling missing dates...")
        self.df = self.cleaner.fill_missing_calendar_days_cross_merge(
            self.df, 
            date_col="date", 
            group_cols=["store_nbr", "item_nbr"]
        )
        return self.df
    
    # 4. Fill NaN in 'onpromotion' with False (or 0 if numeric)
    def fill_missing_onpromotion(self):
        if "onpromotion" in self.df.columns:
            print("   • Filling NaN in 'onpromotion' with False")
            self.df = self.cleaner.fill_nan(
                self.df, 
                column="onpromotion", 
                fill_value=False, 
                cast_type=bool)
    
    
    # 5. Show descriptive statistics
    def run_descriptive_statistics(self):
        print("➡ Running descriptive statistics...")
        desc_stats = self.cleaner.show_description(self.df)
        self.report["descriptive_statistics"] = desc_stats
        display(desc_stats)
        return desc_stats
    
    # 6. Handle negative sales
    def handle_negative_sales(self):
        if (self.df["unit_sales"] < 0).any():
            print("   • Handling negative sales by setting them to zero")
            self.df = self.cleaner.plot_outliers_before_after(self.df, method="negative")
            
    # 7. Distribution Analysis (optimized: use sample for visualization)
    def run_distribution_analysis(self):
        print("➡ Running distribution analysis...")
        
        # For 34M rows, sample 100k for distribution plot
        sample_size = min(100000, len(self.df))
        df_sample = self.df.sample(n=sample_size, random_state=42)
        
        plt.figure(figsize=(12, 6))
        sns.histplot(df_sample["unit_sales"], kde=True)
        plt.title(f"Unit Sales Distribution (n={sample_size:,} sample)")
        plt.savefig(os.path.join(self.eda_path, "unit_sales_distribution.png"))
        plt.show()
        print(f"   • Distribution plot created using {sample_size:,} sample")
        print(f"   • Distribution plot saved to {self.eda_path}/unit_sales_distribution.png")

    # 8. Visual Time Series EDA
    def run_time_series_visual_eda(self):
        print("➡ Running visual time-series EDA...")
        self.visualizer.plot_total_sales_over_time()
        #self.visualizer.plot_monthly_sales_heatmap()
        #self.visualizer.plot_sales_by_day_of_week()
        #self.visualizer.plot_monthly_boxplot()
        self.visualizer.plot_sales_by_store()
        #self.visualizer.plot_year_over_year()
        self.visualizer.plot_autocorrelation()

    # 9. Statistical Diagnostics
    def run_time_series_diagnostics(self):
        print("➡ Running statistical time-series diagnostics...")
        self.diagnostics.plot_autocorrelation()
        self.diagnostics.plot_stationarity_checks()
        adf_results = self.diagnostics.run_adf_test()
        stl_res = self.diagnostics.plot_stl_decomposition()
        strengths = self.diagnostics.calculate_strengths(stl_res)
        self.report["diagnostics"] = {"adf": adf_results, "strengths": strengths}
        return self.report["diagnostics"]

    # 10. Transformation Analysis (optimized: use sample)
    def run_transformation_analysis(self):
        print("➡ Running transformation analysis...")
        
        # Use full data for statistics (fast operations)
        stats = {
            "original_skew": self.df["unit_sales"].skew(),
            "original_kurtosis": self.df["unit_sales"].kurtosis()
        }
        
        # Sample for transformed stats (if needed for visualization)
        sample_size = min(1000000, len(self.df))
        df_sample = self.df.sample(n=sample_size, random_state=42)
        df_transformed = self.cleaner.log_transform_sales(df_sample, column="unit_sales")
        
        stats["transformed_skew"] = df_transformed["unit_sales_log"].skew()
        stats["transformed_kurtosis"] = df_transformed["unit_sales_log"].kurtosis()
        
        self.report["transformations"] = stats
        display(pd.DataFrame([stats]))
        return stats

    # 11. Outlier Analysis (optimized: use sample for visualization)
    def run_outlier_report(self, method="zscore", **kwargs):
        """
        Run outlier detection and handling.
        
        Args:
            method (str): Outlier handling method. Options:
                - "negative": Remove negative sales
                - "fixed": Clip to fixed upper value (requires 'upper' kwarg)
                - "zscore": Z-score based clipping (default threshold=3)
                - "iqr": IQR-based clipping (default k=1.5)
                - "logclip": Log-transform + percentile clipping (default clip_percentile=0.99)
                - "logzscore": Log-transform + Z-score (robust combined approach) [RECOMMENDED]
            **kwargs: Additional parameters for the chosen method:
                - threshold: Z-score threshold (default=3.0)
                - k: IQR multiplier (default=1.5)
                - upper: Fixed upper bound for "fixed" method
                - clip_percentile: Percentile for "logclip" method (default=0.99)
        """
        print("➡ Running outlier report...")
        print(f"   • Outlier handling method: {method}")
        
        # Set default parameters based on method
        if method == "zscore" and "threshold" not in kwargs:
            kwargs["threshold"] = 6.0
        elif method == "iqr" and "k" not in kwargs:
            kwargs["k"] = 5.0
        elif method == "logclip" and "clip_percentile" not in kwargs:
            kwargs["clip_percentile"] = 0.99
        elif method == "logzscore" and "threshold" not in kwargs:
            kwargs["threshold"] = 3.0
        
        # Print method description
        method_descriptions = {
            "negative": "Replace negative sales (returns) with 0",
            "fixed": "Clip to fixed upper value",
            "zscore": "Z-score based clipping (global)",
            "iqr": "IQR-based clipping (Q1 - k*IQR, Q3 + k*IQR)",
            "logclip": "Log-transform + percentile clipping",
            "logzscore": "Log-transform + Z-score capping (robust combined approach) [RECOMMENDED]"
        }
        print(f"   • Description: {method_descriptions.get(method, 'Custom method')}")
        print(f"   • Parameters: {kwargs}")
        
        print("   • Using sampled data for outlier visualization...")
        
        
        # Apply to sample first to visualize
        self.df = self.cleaner.plot_outliers_before_after(
            self.df, 
            column="unit_sales", 
            method=method, 
            **kwargs
        )

    def run_advanced_outlier_analysis(self):
        print("➡ Running advanced outlier analysis...")
        outlier_reports = {}
        iqr_outliers = self.cleaner.detect_extreme_sales_iqr(self.df)
        outlier_reports["iqr_count"] = len(iqr_outliers)
        zscore_outliers = self.cleaner.detect_extreme_sales_zscore(self.df)
        outlier_reports["zscore_count"] = len(zscore_outliers)
        self.report["outlier_analysis"] = outlier_reports
        display(pd.DataFrame([outlier_reports]))
        return outlier_reports

    # 12. Time Series Gap Analysis
    def run_time_series_gap_analysis(self):
        print("➡ Running time series gap analysis...")
        all_dates = pd.date_range(start=self.df["date"].min(), end=self.df["date"].max(), freq='D')
        missing_dates = set(all_dates) - set(self.df["date"].unique())
        gap_report = {
            "total_days_in_range": len(all_dates),
            "days_with_data": self.df["date"].nunique(),
            "missing_days": len(missing_dates),
            "data_coverage": (self.df["date"].nunique() / len(all_dates)) * 100
        }
        self.report["gap_analysis"] = gap_report
        display(pd.DataFrame([gap_report]))
        return gap_report

    # Master Pipeline
    def run_full_eda(self):
        print("====================================================")
        print("🚀 STARTING FULL EDA PIPELINE")
        print("====================================================")

        self.run_structural_summary()
        self.run_data_quality_checks()
        self.run_missing_value_report()
        self.check_and_plot_missing_calendar()
        self.complete_daily_coverage()
        self.fill_missing_onpromotion()
        self.run_descriptive_statistics()
        self.handle_negative_sales()
       
        self.run_distribution_analysis()
        self.run_transformation_analysis()
        self.run_advanced_outlier_analysis()
        self.run_time_series_visual_eda()
        self.run_time_series_diagnostics()
        self.run_time_series_gap_analysis()
        self.run_outlier_report()

        print("====================================================")
        print(f"🎉 EDA COMPLETE! Visuals saved to: {self.eda_path}")
        print(f"📊 Stats saved to: {self.stats_path}")
        print("====================================================")
        self.save_cleaned_data()
        return self.report
    
    def save_cleaned_data(self, filename="train_cleaned.csv"):
        save_path = os.path.join(get_path("cleaned", week=self.week), filename)
        self.df.to_csv(save_path, index=False)
        print(f"💾 Cleaned dataset saved as {save_path}")
    
