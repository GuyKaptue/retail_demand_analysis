# src/core/week_1/processor/features/data_preparation_pipeline.py

import os
from typing import Optional, Dict, Any, List

from IPython.display import display # type: ignore
import numpy as np # type: ignore
import pandas as pd # type: ignore
from .feature_engineering import FeatureEngineering
from .impact_analysis import ImpactAnalysis
from .feature_viz import FeatureViz
from src.utils import get_path

# Plot heatmap
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np  # noqa: F811


class DataPreparationPipeline:
    """
    Notebook-friendly pipeline orchestrating Feature Engineering and Impact Analysis
    for retail time-series forecasting.

    Design:
    - FeatureEngineering and ImpactAnalysis are separate modules
    - Each step can be run independently in a notebook cell
    - Rich debug summaries after each stage
    """

    def __init__(
        self,
        df: pd.DataFrame,
        holidays_df: pd.DataFrame = None,
        oil_df: pd.DataFrame = None,
        date_col: str = "date",
        target_col: str = "unit_sales",
        price_col: str = "unit_price",
        week: int = 1
    ):
        self.df = df.copy()
        self.holidays_df = holidays_df
        self.oil_df = oil_df
        self.date_col = date_col
        self.target_col = target_col
        self.price_col = price_col
        self.week = week
        
        self.merged_holidays_df = None  # To store merged holidays data if needed
        self.merged_oil_df = None       # To store merged oil data if needed

        # Paths
        self.viz_path = get_path("features_viz", week=self.week)
        self.processed_path = get_path("features", week=self.week)
        self.results_stats_path = get_path("features_results", week=self.week)
        os.makedirs(self.viz_path, exist_ok=True)
        os.makedirs(self.processed_path, exist_ok=True)
        os.makedirs(self.results_stats_path, exist_ok=True)

        # Components
        self.feature_engineer = FeatureEngineering(self.df, week=self.week)
        self.impact_analyzer = ImpactAnalysis(self.df, week=self.week)
        self.viz = FeatureViz(df=self.df, week=self.week)

    # ---------------------------------------------------------------------
    # Utility
    # ---------------------------------------------------------------------
    def summary(self, title: str):
        print("="*70)
        print(f"✅ {title}")
        print("="*70)
        print("Summary:")
        print(f"  • Rows: {len(self.df):,}")
        print(f"  • Columns: {self.df.shape[1]}")
        if self.date_col in self.df.columns:
            print(f"  • Date range: {self.df[self.date_col].min()} → {self.df[self.date_col].max()}")
        if "store_nbr" in self.df.columns:
            print(f"  • Unique stores: {self.df['store_nbr'].nunique()}")
        if "item_nbr" in self.df.columns:
            print(f"  • Unique items: {self.df['item_nbr'].nunique()}")
        print(f"  • Memory: {self.df.memory_usage(deep=True).sum() / (1024**2):.2f} MB")
        print("="*70 + "\n")

    #############################
    # Oil Data Cleaning
    #############################

    def clean_oil_data(self, title: str = "Oil Data Cleaning") -> pd.DataFrame:
        """
        Clean oil dataset by converting 'date' to datetime and 
        handling missing values in 'dcoilwtico' using linear interpolation 
        with forward fill and backward fill fallback.

        Parameters:
            title (str): Title for debug summary output.

        Returns:
            pd.DataFrame: Cleaned oil dataset.
        """
        if self.oil_df is None:
            print("[WARN] No oil_df provided; skipping oil data cleaning.")
            return None

        print("="*70)
        print(f"🔧 {title}")
        print("="*70)

        # Convert 'date' to datetime
        self.oil_df['date'] = pd.to_datetime(self.oil_df['date'])

        # Check missing values before
        print("Missing before:", self.oil_df['dcoilwtico'].isna().sum())

        # Apply linear interpolation
        self.oil_df['dcoilwtico'] = self.oil_df['dcoilwtico'].interpolate(method='linear')

        # Apply forward fill and backward fill as fallback (for any remaining NaN at edges)
        self.oil_df['dcoilwtico'] = self.oil_df['dcoilwtico'].ffill().bfill()

        # Check missing values after
        print("Missing after:", self.oil_df['dcoilwtico'].isna().sum())

        print("="*70 + "\n")

        return self.oil_df

    
    # ---------------------------------------------------------------------
    # Feature Engineering Methods (delegated)
    # ---------------------------------------------------------------------
    def add_date_features(self):
        self.feature_engineer.add_date_features(date_col=self.date_col)
        self.df = self.feature_engineer.df
        self.summary("Date Features Added")
    
    def plot_calendar_trends(self):
        self.feature_engineer.plot_calendar_trends()
        self.summary("Calendar trends plots")

    def add_target_transform(self):
        self.feature_engineer.add_target_transform(sales_col=self.target_col)
        self.df = self.feature_engineer.df
        self.summary("Target Transform Added")

    def add_advanced_lags(self, lags=(7, 14, 28, 365)):
        self.feature_engineer.add_advanced_lags(sales_col=self.target_col, lags=lags)
        self.df = self.feature_engineer.df
        self.summary("Advanced Lags Added")

    def add_rolling_features(self, windows=(7, 28, 365), stats=("mean", "std")):
        self.feature_engineer.add_rolling_features(
            sales_col=self.target_col, windows=windows, stats=stats
        )
        self.df = self.feature_engineer.df
        self.summary("Rolling Features Added")
        
    def add_rolling_smoothing(
        self,
        metrics=["unit_sales"],
        windows=[3, 7, 14, 30, 365],
        stats=["mean", "std", "median"],   # any rolling stat
        group_cols=["item_nbr", "store_nbr"]
    ):
        self.feature_engineer.add_rolling_smoothing(
            metrics=metrics,
            windows=windows,
            stats=stats,
            group_cols=group_cols
            )
        self.df = self.feature_engineer.df
        self.summary("Rolling smootthing Features Added")
        
    def plot_rolling_smoothing_trends(self, method="mean"):
        self.feature_engineer.plot_rolling_smoothing_trends(method=method)
        self.summary(f"Rolling smootthing Features with with stats {method} plotting ")
        
    def add_promotion_features(self, promo_col="onpromotion", promo_lags=(1, 7)):
        self.feature_engineer.add_promotion_features(promo_col=promo_col, promo_lags=promo_lags)
        self.df = self.feature_engineer.df
        self.summary("Promotion Features Added")

    def add_price_features(self):
        if self.price_col in self.df.columns:
            self.feature_engineer.add_price_features(price_col=self.price_col)
            self.df = self.feature_engineer.df
            self.summary("Price Features Added")
        else:
            print(f"[WARN] Price column '{self.price_col}' not found in DataFrame.")

    def add_holiday_distance(self):
        if self.holidays_df is not None:
            self.feature_engineer.add_holiday_distance(
                holidays_df=self.holidays_df, date_col=self.date_col
            )
            self.df = self.feature_engineer.df
            self.summary("Holiday Distance Features Added")
        else:
            print("[WARN] No holidays_df provided; skipping holiday distance features.")

    def add_store_item_aggregates(self):
        self.feature_engineer.add_store_item_aggregates(sales_col=self.target_col)
        self.df = self.feature_engineer.df
        self.summary("Store & Item Aggregates Added")

    # ---------------------------------------------------------------------
    # Feature Engineering (separate)
    # ---------------------------------------------------------------------
    def run_feature_engineering(self):
        """Run feature engineering only (cell by cell in notebooks)."""
        print("🚀 Starting Feature Engineering...")

        self.feature_engineer.add_date_features(date_col=self.date_col)
        self.feature_engineer.add_target_transform(sales_col=self.target_col)
        self.feature_engineer.add_advanced_lags(sales_col=self.target_col, lags=(7, 28, 365))
        self.feature_engineer.add_rolling_features(sales_col=self.target_col, windows=(7, 28), stats=("mean", "std"))
        self.feature_engineer.add_promotion_features(promo_col="onpromotion", promo_lags=(1, 7))

        if self.price_col in self.df.columns:
            self.feature_engineer.add_price_features(price_col=self.price_col)

        if self.holidays_df is not None:
            self.feature_engineer.add_holiday_distance(self.holidays_df, date_col=self.date_col)

        self.feature_engineer.add_store_item_aggregates(sales_col=self.target_col)

        # Update df
        self.df = self.feature_engineer.df

        # Save intermediate
        path = os.path.join(self.processed_path, "feature_engineered.csv")
        self.df.to_csv(path, index=False)
        print(f"💾 Feature engineered dataset saved → {path}")

        self.summary("Feature Engineering Complete")

    # ---------------------------------------------------------------------
    # Impact Analysis (separate)
    # ---------------------------------------------------------------------
    def run_impact_analysis(self):
        """Run impact analysis only (cell by cell in notebooks)."""
        if self.oil_df is None and self.holidays_df is None:
            print("⚠️ Skipping Impact Analysis — no external data supplied.")
            return

        print(" Starting Impact Analysis...")

        # Sync latest dataframe
        self.impact_analyzer.df_sales = self.df.copy()
        print(f"[DEBUG] df_sales synced. Shape: {self.impact_analyzer.df_sales.shape}")

        # -------------------------------
        # Oil analysis
        # -------------------------------
        if self.oil_df is not None:
            print(f"[DEBUG] Running Oil Price Impact. Oil DF shape: {self.oil_df.shape}")
            self.merged_oil_df = self.impact_analyzer.merge_oil_prices(self.oil_df, date_col=self.date_col)
            print(f"[DEBUG] Merged oil dataframe shape: {self.merged_oil_df.shape}")
            self.impact_analyzer.plot_oil_vs_sales(self.merged_oil_df, date_col=self.date_col)
            print("[DEBUG] Oil vs sales plot completed.")

        # -------------------------------
        # Holiday analysis
        # -------------------------------
        if self.holidays_df is not None:
            print(f"[DEBUG] Running Holiday Impact. Holidays DF shape: {self.holidays_df.shape}")
            self.merged_holidays_df = self.impact_analyzer.merge_holidays(self.holidays_df, date_col=self.date_col)
            print(f"[DEBUG] Merged holiday dataframe shape: {self.merged_holidays_df.shape}")

            self.impact_analyzer.plot_holiday_vs_nonholiday_sales_dual(self.merged_holidays_df, sales_col=self.target_col)
            print("[DEBUG] Holiday impact analysis completed.")

            save_path = self.impact_analyzer.save_merged(self.merged_holidays_df, "sales_with_holidays.csv")
            print(f"[DEBUG] Merged holiday dataset saved → {save_path}")

        self.summary("Impact Analysis Complete")
        print("[DEBUG] run_impact_analysis finished.")
    
    
    


    # ---------------------------------------------------------------------
    # Full Pipeline (optional)
    # ---------------------------------------------------------------------
    def run_pipeline(self):
        """Run both feature engineering and impact analysis together."""
        print("🚀 Running Full Data Preparation Pipeline...")
        self.run_feature_engineering()
        self.run_impact_analysis()

        final_path = os.path.join(self.processed_path, "final_dataset.csv")
        self.df.to_csv(final_path, index=False)
        print(f"🎉 Pipeline completed. Final dataset saved → {final_path}")

        self.summary("PIPELINE COMPLETE")
        return self.df
    
    
    # src/core/week_1/processor/features/feature_engineering.py (or DataPreparationPipeline)

    def optimize_memory(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Downcasts numeric columns to the smallest possible type (int8/16/32, float16/32)
        to reduce memory and disk usage significantly.
        """
        print("🧠 Optimizing DataFrame memory usage...")
        for col in df.columns:
            col_type = df[col].dtype
            
            # Skip objects (like 'date' if not yet converted)
            if col_type != object:  # noqa: E721
                
                # Integer conversion
                if 'int' in str(col_type):
                    c_min = df[col].min()
                    c_max = df[col].max()
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                
                # Float conversion
                elif 'float' in str(col_type):
                    # Use float32 to preserve reasonable precision
                    df[col] = df[col].astype(np.float32)

        new_memory = df.memory_usage(deep=True).sum() / (1024**2)
        print(f"  • New Memory Usage: {new_memory:.2f} MB")
        return df
    # ======================================
    # Numeric Feature Descriptive Statistics
    # ======================================

    def numeric_describe(
        self,
        exclude: Optional[List[str]]= ["store_nbr", "item_nbr", "date", "id"]
    ):
        exclude = exclude or []

        print(" [DEBUG] Starting numeric_describe")
        print(f" [DEBUG] Initial dataframe shape: {self.df.shape}")
        print(f" [DEBUG] Columns to exclude: {exclude}")

        df_filtered = self.df.drop(columns=exclude, errors="ignore")
        print(f" [DEBUG] Shape after exclusion: {df_filtered.shape}")

        numeric_df = df_filtered.select_dtypes(include="number")
        print(f" [DEBUG] Numeric columns selected ({len(numeric_df.columns)}):")
        print(list(numeric_df.columns))

        result = numeric_df.describe().T
        print(f" [DEBUG] Describe output shape: {result.shape}")

        return result
    
    # ======================================
    # Correlation Matrix & Heatmap
    # ======================================

    def correlation_matrix(
        self,
        exclude: Optional[List[str]] = None,
        figsize=(20, 15),
        cmap="coolwarm",
        mask_upper=True,
    ):
        """
        Compute correlation matrix for numeric features (excluding specified columns)
        and plot as a heatmap.

        Parameters:
            exclude (list): Columns to exclude from correlation analysis.
            figsize (tuple): Size of the heatmap figure.
            cmap (str): Colormap for the heatmap.
            mask_upper (bool): Whether to mask the upper triangle of the heatmap.
        """
        print("[DEBUG] Starting correlation_matrix")

        if exclude is None:
            exclude = ["store_nbr", "item_nbr", "date", "id"]

        df_filtered = self.df.drop(columns=exclude, errors="ignore")
        numeric_df = df_filtered.select_dtypes(include="number")

        print(f"[DEBUG] Numeric columns selected: {len(numeric_df.columns)}")

        corr = numeric_df.corr()

        print("=" * 70)
        print("✅ Correlation Matrix Computed")
        print("=" * 70)
        display(corr.head())
        print("=" * 70 + "\n")

        plt.figure(figsize=figsize)

        mask = None
        if mask_upper:
            mask = np.triu(np.ones_like(corr, dtype=bool))

        sns.heatmap(
            corr,
            cmap=cmap,
            annot=False,
            center=0,
            linewidths=0.5,
            mask=mask,
            cbar_kws={"shrink": 0.8},
        )
        plt.title("Feature Correlation Heatmap", fontsize=14)
        plt.tight_layout()

        save_path = os.path.join(self.viz_path, "correlation_heatmap.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()
        plt.close()

        print(f"💾 Correlation heatmap saved → {save_path}")

        return corr


        
        # ======================================
        # Holiday Feature Descriptive Statistics
        # ======================================
        
        
    def holiday_describe(self) -> Optional[Dict[str, Any]]:
        """
        Provide descriptive statistics for the merged holidays DataFrame.
        Summarizes counts, unique values, and distributions for categorical columns,
        with detailed debug output.
        """

        print("🔍 Starting holiday_describe...")

        # 1️⃣ Validate presence of merged_holidays_df
        if getattr(self, "merged_holidays_df", None) is None:
            print("[WARN] No merged_holidays_df provided; skipping holiday describe.")
            return None

        df = self.merged_holidays_df.copy()

        print(f"   ➤ Holidays dataframe shape: {df.shape}")
        print(f"   ➤ Columns: {list(df.columns)}\n")

        # 2️⃣ Ensure date column exists and is datetime
        if "date" not in df.columns:
            print("[WARN] 'date' column missing; cannot compute date range.")
            date_range = "N/A"
        else:
            print("   ➤ Converting 'date' column to datetime (if needed)...")
            if not np.issubdtype(df["date"].dtype, np.datetime64):
                df["date"] = pd.to_datetime(df["date"], errors="coerce")

            min_date = df["date"].min()
            max_date = df["date"].max()
            date_range = f"{min_date} → {max_date}"
            print(f"   ✔ Date range: {date_range}\n")

        # 3️⃣ Build summary dictionary
        print("3️⃣ Computing summary statistics...")

        summary = {
            "rows": len(df),
            "columns": df.shape[1],
            "date_range": date_range,
            "unique_types": df["type"].value_counts().to_dict() if "type" in df.columns else {},
            "unique_locales": df["locale"].value_counts().to_dict() if "locale" in df.columns else {},
            "transferred_counts": df["transferred"].value_counts().to_dict() if "transferred" in df.columns else {},
        }

        # 4️⃣ Full descriptive statistics
        print("4️⃣ Generating full descriptive statistics (include='all')...\n")
        desc = df.describe(include="all").T
        print(desc)

        # 5️⃣ Final summary printout
        print("\n" + "=" * 70)
        print("✅ Holiday Describe Summary")
        print("=" * 70)
        print(f"  • Rows: {summary['rows']:,}")
        print(f"  • Columns: {summary['columns']}")
        print(f"  • Date range: {summary['date_range']}")
        print(f"  • Types: {summary['unique_types']}")
        print(f"  • Locales: {summary['unique_locales']}")
        print(f"  • Transferred flag counts: {summary['transferred_counts']}")
        print("=" * 70 + "\n")

        return summary




    def save_final_dataset(self, filename: str = "final_train_dataset.csv"):
        """Save the final prepared dataset to CSV."""
        
        # *** NEW STEP: Optimize memory before saving ***
        # Delegate the optimization to the feature_engineer or define it here.
        # Assuming we add optimize_memory to DataPreparationPipeline for simplicity:
        self.df = self.optimize_memory(self.df) 
        # **********************************************
        
        path = os.path.join(self.processed_path, filename)
        
        # This is where the OSError occurred, now the DataFrame is smaller.
        self.df.to_csv(path, index=False) 
        
        print(f"💾 Final dataset saved → {path}")