# src/core/week_1/processor/eda/missing_value_and_outlier_handler.py

import os
import pandas as pd  # type: ignore

import matplotlib.pyplot as plt  # type: ignore
import seaborn as sns  # type: ignore
import numpy as np  # type: ignore
from scipy import stats  # type: ignore # make sure to import at the top
from IPython.display import display  # type: ignore # for showing full tables in notebooks
from src.utils import get_path


class RetailDataCleaner:
    """
    Utility class for preparing and cleaning the Favorita Grocery Sales Forecasting dataset.

    Responsibilities:
      • Handle missing values (fill NaNs, visualize missingness, fill missing calendar dates).
      • Detect and correct outliers (negative sales, extreme spikes via Z-score/IQR, clipping).
      • Apply transformations (log scaling and reverse transforms).
      • Visualize distributions before and after outlier handling.
    """

    def __init__(self):
        self.eda_path = get_path("eda", week=1)
        os.makedirs(self.eda_path, exist_ok=True)
        print(f"[DEBUG] Initialized MissingValueAndOutlierHandler with EDA path {self.eda_path}")
        
    
    # ---------------------------
    # Data Inspection Utilities
    # ---------------------------
    def show_table(self, df: pd.DataFrame, n: int = 5) -> None:
        """Display the first n rows of the DataFrame in notebook style."""
        print(f"[DEBUG] Method: show_table → Displaying first {n} rows of DataFrame (shape={df.shape})")
        display(df.head(n))

    def show_dtypes(self, df: pd.DataFrame) -> None:
        """Show column names and their data types."""
        print("[DEBUG] Method: show_dtypes → Displaying column data types")
        dtypes_info = pd.DataFrame({
            "Column": df.columns,
            "Dtype": df.dtypes.values
        })
        display(dtypes_info)

    def show_description(self, df: pd.DataFrame) -> None:
        """
        Show descriptive statistics only for numeric columns.
        """
        print("[DEBUG] Method: show_description → Displaying descriptive statistics for numeric columns")
        if "id" in df.columns:
            print("[DEBUG] Method: remove_id_column → Removing 'id' column")
        df = df.drop(columns=["id"])
        # Use pandas describe with include=[np.number] to restrict to numeric columns
        desc = df.describe(include=[np.number]).T
        
        # Display nicely in notebook
        display(desc)

    # ---------------------------
    # Step 1: Missing Value Handling
    # ---------------------------
    def fill_nan(self, df: pd.DataFrame, column: str, fill_value, cast_type=None) -> pd.DataFrame:
        print(f"[DEBUG] Method: fill_nan → Fill NaN values in a column '{column}' with '{fill_value}' and optionally cast type to {cast_type}")
        if column not in df.columns:
            print(f"❌ Column '{column}' not found in DataFrame.")
            return df
        missing_before = df[column].isna().sum()
        print(f"[DEBUG] Missing before: {missing_before}")
        df[column] = df[column].fillna(fill_value)
        if cast_type is not None:
            df[column] = df[column].astype(cast_type)
            print(f"[DEBUG] Column '{column}' casted to {cast_type.__name__}")
        missing_after = df[column].isna().sum()
        print(f"[DEBUG] Missing after: {missing_after}")
        return df

    def plot_missing_values(self, df: pd.DataFrame) -> None:
        print("[DEBUG] Method: plot_missing_values → Visualize missing values with heatmap and bar chart")
        nulls = pd.DataFrame({
            "missing_count": df.isnull().sum(),
            "missing_percent": (df.isnull().sum() / len(df)) * 100
        })
        nulls = nulls[nulls["missing_count"] > 0]
        
        if nulls.empty:
            print("🎉 No missing values in dataset!")
            return
        
        
        fig, axes = plt.subplots(1, 2, figsize=(16, max(6, len(nulls) * 0.3)))
        sns.heatmap(nulls[["missing_percent"]].T, cmap="Reds", annot=True, fmt=".1f", cbar=True, ax=axes[0])
        axes[0].set_title("Feature Missingness (%)")
        nulls_sorted = nulls["missing_count"].sort_values(ascending=False)
        nulls_sorted.plot(kind="bar", color="salmon", ax=axes[1])
        axes[1].set_title("Missing Values per Column")
        plt.tight_layout()
        save_path = os.path.join(self.eda_path, 'missing_value.png')
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"💾 Missing value plot saved to {save_path}")
        plt.show()
        
        return nulls

    def fill_missing_dates(self, df: pd.DataFrame, date_col="date", group_cols=["store_nbr", "item_nbr"]) -> pd.DataFrame:
        print("[DEBUG] Method: fill_missing_dates → Fill missing calendar dates with 0 sales per store-item")
        df[date_col] = pd.to_datetime(df[date_col])
        def fill_calendar(group):
            g = group.set_index(date_col).sort_index()
            g = g.asfreq("D", fill_value=0)
            for col in group_cols:
                g[col] = group[col].iloc[0]
            return g.reset_index()
        df = df.groupby(group_cols, group_keys=False).apply(fill_calendar)
        print(f"[DEBUG] After filling, DataFrame shape={df.shape}")
        return df
    
    
    # ---------------------------
    # Step 2: Outlier Detection & Removal
    # ---------------------------
    def handle_negative_sales(self, df: pd.DataFrame, column: str = "unit_sales") -> pd.DataFrame:
        print("[DEBUG] Method: handle_negative_sales → Replace negative sales (returns) with 0")
        negatives_before = (df[column] < 0).sum()
        print(f"[DEBUG] Negatives before: {negatives_before}")
        df[column] = df[column].apply(lambda x: max(x, 0))
        negatives_after = (df[column] < 0).sum()
        print(f"[DEBUG] Negatives after: {negatives_after}")
        return df

    def detect_extreme_sales_zscore(self, df: pd.DataFrame, column="unit_sales", group_cols=["store_nbr","item_nbr"], threshold=5.0) -> pd.DataFrame:
        print("[DEBUG] Method: detect_extreme_sales_zscore → Detect extreme spikes using Z-score per group")
        def calc(group):
            mean, std = group[column].mean(), group[column].std()
            group["z_score"] = (group[column] - mean) / (std if std != 0 else 1)
            return group
        df_grouped = df.groupby(group_cols).apply(calc).reset_index(drop=True)
        outliers = df_grouped[df_grouped["z_score"] > threshold]
        print(f"[DEBUG] Outliers detected (Z>{threshold}): {len(outliers)}")
        return outliers

    def detect_extreme_sales_iqr(self, df: pd.DataFrame, column="unit_sales") -> pd.DataFrame:
        print("[DEBUG] Method: detect_extreme_sales_iqr → Detect outliers using IQR method (global)")
        Q1, Q3 = df[column].quantile(0.25), df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower, upper = Q1 - 1.5*IQR, Q3 + 1.5*IQR
        print(f"[DEBUG] IQR bounds: lower={lower}, upper={upper}")
        outliers = df[(df[column] < lower) | (df[column] > upper)]
        print(f"[DEBUG] Outliers detected: {len(outliers)}")
        return outliers

    
    # -----------------------------
    # Outlier Clipping Function
    # -----------------------------
    
    def clip_outlier(
        self,
        df: pd.DataFrame,
        col: str,
        method: str = "fixed",
        upper: float = None,
        threshold: float = 3.0,
        k: float = 1.5,
        clip_percentile: float = 0.99
    ) -> pd.DataFrame:
        """
        Clip outliers in a column using different methods:
        - method="fixed": clip to a fixed upper value (requires 'upper')
        - method="zscore": clip based on Z-score threshold (default threshold=3)
        - method="iqr": clip based on IQR multiplier (default k=1.5)
        - method="logclip": apply log-transform + clip at chosen percentile (default 99th)
        - method="logzscore": apply log-transform + Z-score capping (robust combined approach)

        Advanced Analysis Notes:
        • The IQR method (global) is ineffective for `unit_sales` because the IQR collapses to [0.0, 0.0],
        misclassifying nearly half the dataset as outliers due to dominance of zeros.
        • The Z-score method (group-specific) is more effective, flagging ~282k genuine spikes (Z > 5.0).
        • A combined approach — log transformation followed by Z-score capping — is recommended to
        address both skewness and extreme spikes robustly.
        """

        print(f"[DEBUG] Method: clip_outlier → Clipping outliers in '{col}' using method='{method}'")
        df = df.copy()

        if method == "fixed":
            if upper is None:
                raise ValueError("Für method='fixed' muss ein upper-Wert angegeben werden.")
            print(f"[DEBUG] Fixed clipping: upper={upper}")
            df[col] = df[col].clip(upper=upper)

        elif method == "zscore":
            print(f"[DEBUG] Z-score clipping: threshold={threshold}")
            z_scores = np.abs(stats.zscore(df[col], nan_policy='omit'))
            mask = z_scores > threshold
            upper_bound = df.loc[~mask, col].max()
            print(f"[DEBUG] Computed upper bound (zscore)={upper_bound}")
            df[col] = df[col].clip(upper=upper_bound)

        elif method == "iqr":
            print(f"[DEBUG] IQR clipping: k={k}")
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound, upper_bound = Q1 - k * IQR, Q3 + k * IQR
            print(f"[DEBUG] Computed IQR bounds: lower={lower_bound}, upper={upper_bound}")
            df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)

        elif method == "logclip":
            print(f"[DEBUG] Log+Clip: applying log-transform + clipping at {clip_percentile*100:.1f}th percentile")
            log_col = f"{col}_log"
            df[log_col] = np.log1p(df[col])
            upper_bound = df[log_col].quantile(clip_percentile)
            df[log_col] = df[log_col].clip(upper=upper_bound)
            df[col] = np.expm1(df[log_col])
            print(f"[DEBUG] Combined log-transform + clipping applied. New max={df[col].max()}")

        elif method == "logzscore":
            print(f"[DEBUG] Log+Z-score clipping: threshold={threshold}")
            log_col = f"{col}_log"
            df[log_col] = np.log1p(df[col])
            z_scores = np.abs(stats.zscore(df[log_col], nan_policy='omit'))
            mask = z_scores > threshold
            upper_bound = df.loc[~mask, log_col].max()
            print(f"[DEBUG] Computed upper bound (log-zscore)={upper_bound}")
            df[log_col] = df[log_col].clip(upper=upper_bound)
            df[col] = np.expm1(df[log_col])
            print(f"[DEBUG] Combined log-transform + Z-score clipping applied. New max={df[col].max()}")

        else:
            raise ValueError("Ungültige Methode. Wähle 'fixed', 'zscore', 'iqr', 'logclip' oder 'logzscore'.")

        print(f"[DEBUG] Clipping complete for '{col}'. New max={df[col].max()}")
        return df

    # ---------------------------
    # Step 3: Transformations
    # ---------------------------
    def log_transform_sales(self, df: pd.DataFrame, column="unit_sales") -> pd.DataFrame:
        print("[DEBUG] Method: log_transform_sales → Apply log(1+x) transform to compress large values")
        new_col = f"{column}_log"
        df[new_col] = np.log1p(df[column])
        print(f"[DEBUG] New column '{new_col}' created")
        return df

    def reverse_log_transform(self, df: pd.DataFrame, column="unit_sales_log") -> pd.DataFrame:
        print("[DEBUG] Method: reverse_log_transform → Reverse log transform back to original scale")
        orig_col = column.replace("_log", "")
        df[orig_col + "_reversed"] = np.expm1(df[column])
        print(f"[DEBUG] New column '{orig_col}_reversed' created")
        return df
    
    # ==============================
    # Step 4: Fill Missing Calendar Days
    # ==============================
    
    # src/core/week_1/processor/features/feature_engineering.py (inside FeatureEngineering class)

    # ---------------------------
    # Data Densification (Cross-Merge Method)
    # ---------------------------
    def fill_missing_calendar_days_cross_merge(self, 
                                              date_col: str = "date",
                                              group_cols: list = ["store_nbr", "item_nbr"],
                                              sales_col: str = "unit_sales") -> None:
        """
        Ensures complete daily coverage for each (store, item) pair using the 
        memory-intensive but explicit Cross-Merge method. Missing calendar days are 
        filled with unit_sales = 0.
        
        Args:
            date_col (str): Name of the date column.
            group_cols (list): Columns defining the unique sales series (store, item).
            sales_col (str): Sales column to fill with zeros.
        """
        print("[DEBUG] Method: fill_missing_calendar_days_cross_merge → Starting explicit cross-merge densification.")

        # 1. Ensure datetime format
        self.df[date_col] = pd.to_datetime(self.df[date_col])

        # 2. Define the full date range (T_days)
        min_date = self.df[date_col].min()
        max_date = self.df[date_col].max()
        full_date_range = pd.DataFrame({date_col: pd.date_range(min_date, max_date, freq='D')})
        print(f"[DEBUG] Full date range created: {min_date} to {max_date} ({len(full_date_range)} days)")

        # 3. Define all unique (store, item) combinations (N_groups)
        store_item_combinations = self.df[group_cols].drop_duplicates().reset_index(drop=True)
        print(f"[DEBUG] Unique {group_cols} combinations found: {len(store_item_combinations):,}")

        # 4. Create the Cartesian Product (N_groups * T_days)
        # This creates the massive index of ALL possible (store, item, date)
        print("[DEBUG] Creating Cartesian product (cross merge)...")
        all_combinations = store_item_combinations.merge(full_date_range, how='cross')
        
        # 5. Merge with original data (Left Merge)
        # Keeps all rows from all_combinations and adds sales data where available.
        df_filled = all_combinations.merge(self.df, on=group_cols + [date_col], how='left')

        # 6. Fill missing values (NaN) with 0
        original_n_rows = len(self.df)
        missing_sales = df_filled[sales_col].isna().sum()
        df_filled[sales_col] = df_filled[sales_col].fillna(0)
        
        # 7. Update the class DataFrame
        self.df = df_filled
        
        print(f"[DEBUG] Original rows: {original_n_rows:,}")
        print(f"[DEBUG] Rows filled (NaNs converted to 0): {missing_sales:,}")
        print(f"[DEBUG] Final (dense) DataFrame shape: {self.df.shape}")
        
        display(self.df[group_cols + [date_col, sales_col]].sample(15))
        return df_filled
    
    def fill_missing_calendar_days(self, df: pd.DataFrame,
                               date_col: str = "date",
                               group_cols: list = ["store_nbr", "item_nbr"],
                               sales_col: str = "unit_sales") -> pd.DataFrame:
        """
        Ensure complete daily coverage for each (store, item) pair.
        Missing calendar days are filled with unit_sales = 0.

        Args:
            df (pd.DataFrame): Input DataFrame with sales records.
            date_col (str): Name of the date column.
            group_cols (list): Columns to group by (store, item).
            sales_col (str): Sales column to fill with zeros.

        Returns:
            pd.DataFrame: DataFrame with complete daily index per group.
        """
        # Debug message for traceability
        print("[DEBUG] Method: fill_missing_calendar_days → Creating complete daily index per (store, item)")

        # 1) Ensure the date column is in datetime dtype for reliable indexing and range creation
        df[date_col] = pd.to_datetime(df[date_col])

        # 2) Compute a global full date range from the min to max date present in the entire dataframe
        #    Note: this will be used for reindexing every group to the same calendar span.
        full_range = pd.date_range(df[date_col].min(), df[date_col].max(), freq="D")

        # 3) Define a helper function to reindex a single group (store_nbr, item_nbr)
        def reindex_group(group):
            # a) Set the group’s date as index and sort by date to avoid mis-ordered rows
            g = group.set_index(date_col).sort_index()
            # b) Reindex to the global full_range, filling missing dates with 0 in all columns
            #    (including sales_col); this ensures calendar completeness per group.
            g = g.reindex(full_range, fill_value=0)
            # c) Name the index back as the date column for consistency
            g.index.name = date_col
            # d) Preserve group identifiers by copying the group's keys to each row
            for col in group_cols:
                g[col] = group[col].iloc[0]
            # e) Return the group back to a regular dataframe with date as a column
            return g.reset_index()

        # 4) Apply the reindexing function to each group, concatenating the results
        #    group_keys=False ensures the original index is not added as a hierarchy level.
        df_filled = df.groupby(group_cols, group_keys=False).apply(reindex_group)

        # 5) Print a brief debug summary of the resulting shape for quick sanity-checks
        print(f"[DEBUG] After filling calendar days, DataFrame shape={df_filled.shape}")

        # 6) Show a small random sample to visually verify the zero-filling and identifiers
        display(df_filled.sample(15))

        # 7) Return the calendar-complete dataframe
        return df_filled


    
    # ---------------------------
    # Step 5: Outlier Visualization
    # ---------------------------
    def plot_outliers_before_after(
        self,
        df: pd.DataFrame,
        method: str = "fixed",
        column: str = "unit_sales",
        k: float = 5,
        **kwargs) -> pd.DataFrame:
        """
        Plot boxplot + histogram before/after outlier handling
        and automatically save cleaned DataFrame as train_cleaner.csv.
        """
        print(f"[DEBUG] Method: plot_outliers_before_after → Compare distributions before/after handling ({method})")

        # Copy original column for comparison
        before = df[column].copy()

        # ------------------------------------------
        # Apply chosen outlier / negative handling
        # ------------------------------------------
        if method == "negative":
            print("[DEBUG] Handling negative sales...")
            df = self.handle_negative_sales(df, column)

        elif method in ["zscore", "iqr", "clip", "fixed", "logclip", 'logzscore']:
            print(f"[DEBUG] Clipping extreme values using {method} method...")
            df = self.clip_outlier(df, col=column, method=method, **kwargs)

        else:
            raise ValueError("Method must be 'negative', 'zscore', 'iqr', 'clip', 'fixed', 'logclip', or 'logzscore'")

        after = df[column].copy()

        print(f"[DEBUG] Plotting distributions before vs after using method '{method}'")

        # ------------------------------------------
        # Visualization
        # ------------------------------------------
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        sns.boxplot(x=before, ax=axes[0, 0], color="skyblue")
        axes[0, 0].set_title("Boxplot Before Outlier Handling")

        sns.histplot(before, kde=True, ax=axes[1, 0], color="skyblue")
        axes[1, 0].set_title("Histogram + KDE Before Outlier Handling")

        sns.boxplot(x=after, ax=axes[0, 1], color="lightgreen")
        axes[0, 1].set_title("Boxplot After Outlier Handling")

        sns.histplot(after, kde=True, ax=axes[1, 1], color="lightgreen")
        axes[1, 1].set_title("Histogram + KDE After Outlier Handling")

        plt.tight_layout()

        # Save plot
        plot_path = os.path.join(self.eda_path, f"outliers_{method}_before_after.png")
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        print(f"💾 Outlier comparison plot saved to {plot_path}")
        plt.show()

        # ------------------------------------------
        # NEW: Save cleaned DataFrame as train_cleaned.csv
        # ------------------------------------------
        #save_cleaned_path = os.path.join(get_path("cleaned", week=1), "train_cleaned.csv")
        #df.to_csv(save_cleaned_path, index=False)
        #print(f"💾 Cleaned dataset saved as {save_cleaned_path}")

        return df
