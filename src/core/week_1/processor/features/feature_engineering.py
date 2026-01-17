# src/core/week_1/processor/features/feature_engineering.py


# src/core/week_1/processor/features/feature_engineering.py

import os
from typing import Iterable, Optional, Dict
import pandas as pd # type: ignore
import numpy as np # type: ignore
import matplotlib.pyplot as plt # type: ignore
from IPython.display import display # type: ignore
from sklearn.feature_selection import VarianceThreshold # type: ignore
from statsmodels.stats.outliers_influence import variance_inflation_factor # type: ignore

from src.utils import get_path
from .feature_viz import FeatureViz


class FeatureEngineering:
    """
    Professional FeatureEngineering for retail time-series (Favorita).
    Each feature creation method is grouped with its visualization method(s),
    making it notebook-friendly and transparent.
    """

    def __init__(self, df: pd.DataFrame, week: int = 1):
        self.df = df.copy()
        self.week = week
        self.processed_path = get_path("features", week=self.week)
        self.results_stats_path = get_path("features_results", week=self.week)
        os.makedirs(self.results_stats_path, exist_ok=True)
        os.makedirs(self.processed_path, exist_ok=True)
        print(f"[DEBUG] Initialized FeatureEngineering with DataFrame shape {self.df.shape}")
        self.viz = FeatureViz(df=self.df, week=self.week)

    # ---------------------------
    # Utilities
    # ---------------------------
    def _prepare(self, date_col: str = "date"):
        self.df[date_col] = pd.to_datetime(self.df[date_col])
        self.df = self.df.sort_values(["item_nbr", "store_nbr", date_col]).reset_index(drop=True)
        # Update FeatureViz with the prepared dataframe
        self.viz.df = self.df

    # ---------------------------
    # Date Features + Viz
    # ---------------------------
    def add_date_features(self, date_col: str = "date") -> None:
        """Add date-based features and visualize temporal patterns"""
        self._prepare(date_col)
        
        # Feature creation
        self.df["year"] = self.df[date_col].dt.year
        self.df["month"] = self.df[date_col].dt.month
        self.df["day"] = self.df[date_col].dt.day
        self.df["day_of_week"] = self.df[date_col].dt.dayofweek
        self.df["week_of_year"] = self.df[date_col].dt.isocalendar().week.astype(int)
        self.df["quarter"] = self.df[date_col].dt.quarter
        self.df["is_weekend"] = self.df["day_of_week"].isin([5, 6]).astype(int)
        self.df["day_of_year"] = self.df[date_col].dt.dayofyear
        
        print("[DEBUG] Added basic date features")
        display(self.df[["date", "year", "month", "day", "day_of_week", "week_of_year"]].sample(10))

        # Visualization
        #self.viz.plot_daily_coverage(date_col)
        self.viz.plot_sales_by_day_of_week(sales_col="unit_sales")
        
    # In FeatureEngineering class, replace or add this method:
    def plot_calendar_trends(self, date_col="date", sales_col="unit_sales"):
        self.viz.plot_calendar_feature_trends(date_col=date_col, sales_col=sales_col)

    def plot_monthly_patterns(self):
        """
        Create a 2x2 grid plot showing monthly sales patterns
        Includes: heatmap, total sales, boxplot, and year-over-year comparison
        """
        
        
        # Create figure with 2x2 grid
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Monthly Sales Patterns Analysis', fontsize=16, fontweight='bold', y=1.02)
        
        # Plot 1: Monthly Sales Heatmap
        ax1 = axes[0, 0]
        self.viz._plot_monthly_heatmap(ax=ax1)
        ax1.set_title('Monthly Sales Heatmap', fontweight='bold', pad=15)
        
        # Plot 2: Monthly Total Sales
        ax2 = axes[0, 1]
        self.viz._plot_monthly_total_sales(ax=ax2)
        ax2.set_title('Monthly Total Sales Trend', fontweight='bold', pad=15)
        
        # Plot 3: Monthly Boxplot
        ax3 = axes[1, 0]
        self.viz._plot_monthly_boxplot(ax=ax3)
        ax3.set_title('Monthly Sales Distribution', fontweight='bold', pad=15)
        
        # Plot 4: Year-over-Year Comparison
        ax4 = axes[1, 1]
        self.viz._plot_year_over_year(ax=ax4)
        ax4.set_title('Year-over-Year Comparison', fontweight='bold', pad=15)
        
        # Adjust layout
        plt.tight_layout()
        plt.show()

    # ---------------------------
    # Target Transform + Viz
    # ---------------------------
    def add_target_transform(self, sales_col: str = "unit_sales") -> None:
        """Apply transformations to target variable and visualize distributions"""
        self.df[f"{sales_col}_log"] = np.log1p(self.df[sales_col])
        self.df = self.df.sort_values(["item_nbr", "store_nbr", "date"])
        self.df[f"{sales_col}_pct_change_7"] = (
            self.df.groupby(["item_nbr", "store_nbr"])[sales_col].pct_change(7)
        )
        
        print("[DEBUG] Added target transforms")
        display(self.df[[sales_col, f"{sales_col}_log", f"{sales_col}_pct_change_7"]].sample(10))

        # Visualization
        self.viz.plot_target_distribution(sales_col)
        self.viz.plot_pct_change_distribution(sales_col, window=7)
    
    # ---------------------------
    # Demand Elasticity Features (Internal Logic ONLY)
    # ---------------------------
    def add_demand_elasticity_features(self, sales_col: str = "unit_sales") -> None:
        """
        Creates features to capture demand elasticity based on the 'onpromotion' status,
        serving as a price proxy when 'sell_price' is unavailable.
        Uses only internal dataset values (unit_sales, onpromotion).
        """
        promo_col: str = "onpromotion"

        if promo_col not in self.df.columns:
            print(f"[WARN] Required column '{promo_col}' not in DataFrame; skipping demand elasticity features.")
            return

        self._prepare("date")

        # Temporary sales column to handle returns (negative values)
        temp_sales = self.df[sales_col].clip(lower=0)

        # 1. Calculate Mean Sales during Promotion and Non-Promotion (Per item_nbr)
        # This is the core element for measuring elasticity.

        # Mean sales during promotion (True)
        # Use .loc[] to filter the DataFrame before applying groupby/transform
        promo_sales_mean = self.df.loc[self.df[promo_col] == True].groupby("item_nbr")[temp_sales].transform("mean")  # noqa: E712

        # Mean sales during non-promotion (False)
        non_promo_sales_mean = self.df.loc[self.df[promo_col] == False].groupby("item_nbr")[temp_sales].transform("mean")  # noqa: E712

        # 2. Integrate the Means into the Main DataFrame
        # We need these mean values available across all rows for the item

        # Initialize temporary columns
        self.df["item_avg_sales_promo"] = np.nan
        self.df["item_avg_sales_non_promo"] = np.nan

        # Fill the calculated means back into the temporary columns
        # Use .loc[...].fillna() to ensure the transformed series align correctly
        self.df.loc[self.df[promo_col] == True, "item_avg_sales_promo"] = promo_sales_mean  # noqa: E712
        self.df.loc[self.df[promo_col] == False, "item_avg_sales_non_promo"] = non_promo_sales_mean  # noqa: E712

        # Fill NaNs across all rows with the overall mean for that item (using transform on the non-filtered series)
        self.df["item_avg_sales_promo"] = self.df.groupby("item_nbr")["item_avg_sales_promo"].transform(lambda x: x.fillna(x.mean()))
        self.df["item_avg_sales_non_promo"] = self.df.groupby("item_nbr")["item_avg_sales_non_promo"].transform(lambda x: x.fillna(x.mean()))


        # 3. FEATURE: Promotion Response Ratio (Elasticity)
        # The ratio of sales when on promotion vs. sales when not on promotion.
        # A high value (e.g., 2.0) means sales double when promoted, indicating high price sensitivity.
        self.df["promo_response_ratio"] = self.df["item_avg_sales_promo"] / self.df["item_avg_sales_non_promo"].replace(0, np.nan)

        # 4. FEATURE: Relative Sales Performance (Substitute for 'price_rel_to_item')
        # How do current sales compare to the expected average sales for the *current* promotion state?
        # If currently on promo, compare to item_avg_sales_promo, otherwise use item_avg_sales_non_promo.
        self.df["sales_rel_to_state"] = self.df[sales_col] / np.where(
            self.df[promo_col] == True,  # noqa: E712
            self.df["item_avg_sales_promo"],
            self.df["item_avg_sales_non_promo"]
        ).replace(0, np.nan)

        # 5. FEATURE: Rolling Promotion Frequency (Substitute for 'price_r7_std')
        # The frequency of promotions over the last 28 days (measures consistency of pricing/discounting)
        self.df["promo_r28_freq"] = self.df.groupby(["item_nbr", "store_nbr"])[promo_col].transform(
            lambda s: s.rolling(28, min_periods=1).mean()
        )

        # Clean up the temporary average columns
        self.df = self.df.drop(columns=["item_avg_sales_promo", "item_avg_sales_non_promo"], errors='ignore')

        print("[DEBUG] Added internal demand elasticity features (ratio, frequency, relative sales)")
        display(self.df[[
            sales_col,
            promo_col,
            "promo_response_ratio",
            "sales_rel_to_state",
            "promo_r28_freq"
        ]].dropna().head(3))

        # Visualization
        # Use an existing plot that makes sense for the new frequency feature.
        self.viz.plot_promotion_streak_distribution(promo_freq_col="promo_r28_freq")

    # ---------------------------
    # Lags + Viz
    # ---------------------------
    def add_advanced_lags(self, sales_col: str = "unit_sales", lags: Iterable[int] = (3, 7, 14, 30, 365)) -> None:
        """Create lag features and visualize temporal relationships"""
        self._prepare("date")
        
        # Feature creation
        for lag in lags:
            self.df[f"{sales_col}_lag_{lag}"] = (
                self.df.groupby(["item_nbr", "store_nbr"])[sales_col].shift(lag)
            )
        if 365 in lags:
            self.df[f"{sales_col}_lag_same_dow_last_year"] = (
                self.df.groupby(["item_nbr", "store_nbr"])[sales_col].shift(365)
            )
        
        print(f"[DEBUG] Added lag features: {list(lags)}")
        display(self.df.filter(like=f"{sales_col}_lag_").sample(10))

        # Visualization
        self.viz.plot_lags_overlay(sales_col=sales_col, lags=lags)

    # ---------------------------
    # Rolling Features + Viz
    # ---------------------------
    def add_rolling_features(self, sales_col: str = "unit_sales", 
                             windows: Iterable[int] = (7, 28, 365),
                             stats: Iterable[str] = ("mean", "std")) -> None:
        """Create rolling window features and visualize trends"""
        self._prepare("date")
        df_idx = self.df.set_index("date")
        grouped = df_idx.groupby(["item_nbr", "store_nbr"])[sales_col]
        new_cols: Dict[str, pd.Series] = {}

        for w in windows:
            rolled = grouped.rolling(window=w, min_periods=1)
            for stat in stats:
                colname = f"{sales_col}_r{w}_{stat}"
                ser = getattr(rolled, stat)().reset_index(level=[0, 1], drop=True)
                new_cols[colname] = ser

        for k, v in new_cols.items():
            self.df[k] = v.values

        print(f"[DEBUG] Added rolling features for windows: {list(windows)}, stats: {list(stats)}")
        display(self.df.filter(regex=r"unit_sales_r\d+_").sample(10))

        # Visualization
        self.viz.plot_rolling_mean_std(sales_col=sales_col, window=7)
        
    def add_rolling_smoothing(
        self,
        metrics=["unit_sales"],
        windows=[3, 7, 14, 30, 365],
        stats=["mean", "std", "median"],   # any rolling stat
        group_cols=["item_nbr", "store_nbr"]
    ):
        """
        Unified rolling feature generator:
        - supports multiple metrics
        - supports multiple windows
        - supports multiple stats (mean, std, median, min, max, etc.)
        - consistent naming: metric_r{window}_{stat}
        """

        print("🔧 Starting unified rolling feature generation...")
        print(f"   ➤ Grouping by: {group_cols}")
        print(f"   ➤ Metrics: {metrics}")
        print(f"   ➤ Windows: {windows}")
        print(f"   ➤ Stats: {stats}\n")

        # Validate stats
        valid_stats = {"mean", "std", "median", "min", "max"}
        for s in stats:
            if s not in valid_stats:
                raise ValueError(f"Unsupported stat '{s}'. Allowed: {valid_stats}")

        # Ensure correct ordering
        print("📅 Sorting DataFrame by group columns and date...")
        self.df = self.df.sort_values(group_cols + ["date"]).reset_index(drop=True)
        print("   ✔ Sorting complete.\n")

        # Loop through metrics
        for metric in metrics:
            if metric not in self.df.columns:
                print(f"⚠️ Skipping '{metric}' — column not found.\n")
                continue

            print(f"📊 Processing metric: '{metric}'")

            # Loop through windows
            for w in windows:
                print(f"   ➤ Window: {w} days")

                # Loop through stats
                for stat in stats:
                    colname = f"{metric}_r{w}_{stat}"

                    print(f"      • Creating: {colname}")

                    if stat == "mean":
                        self.df[colname] = (
                            self.df.groupby(group_cols)[metric]
                                .transform(lambda s: s.rolling(w, min_periods=1).mean())
                        )
                    elif stat == "std":
                        self.df[colname] = (
                            self.df.groupby(group_cols)[metric]
                                .transform(lambda s: s.rolling(w, min_periods=1).std())
                        )
                    elif stat == "median":
                        self.df[colname] = (
                            self.df.groupby(group_cols)[metric]
                                .transform(lambda s: s.rolling(w, min_periods=1).median())
                        )
                    elif stat == "min":
                        self.df[colname] = (
                            self.df.groupby(group_cols)[metric]
                                .transform(lambda s: s.rolling(w, min_periods=1).min())
                        )
                    elif stat == "max":
                        self.df[colname] = (
                            self.df.groupby(group_cols)[metric]
                                .transform(lambda s: s.rolling(w, min_periods=1).max())
                        )

                    print(f"        ✔ Done. Sample:\n{self.df[colname].head(3)}\n")

            print(f"✔ Finished metric: {metric}\n")
        print(f"[DEBUG] Added rolling features for windows: {list(windows)}, stats: {list(stats)}")
        # display(self.df.filter(regex=r"unit_sales_r\d+_").sample(10))
        print("🎉 All rolling features generated successfully.\n")
        self.viz = FeatureViz(df=self.df, week=self.week)

    # -------------------------------
    # Rolling Features Visualizations 
    # -------------------------------
    
    def plot_rolling_smoothing_trends(self, method="mean"):
        self.viz.plot_rolling_smoothing_trends(method=method)

    # ---------------------------
    # Promotion Features + Viz
    # ---------------------------
    def add_promotion_features(self, promo_col: str = "onpromotion", 
                               promo_lags: Iterable[int] = (1, 7)) -> None:
        """Create promotion-related features and visualize patterns"""
        self._prepare("date")
        
        # Feature creation
        for lag in promo_lags:
            self.df[f"{promo_col}_lag_{lag}"] = (
                self.df.groupby(["item_nbr", "store_nbr"])[promo_col].shift(lag)
            )

        def streak(s: pd.Series) -> pd.Series:
            return s.groupby((s != s.shift()).cumsum()).cumcount().add(1) * s

        self.df["promo_streak"] = self.df.groupby(["item_nbr", "store_nbr"])[promo_col].transform(streak)
        
        print("[DEBUG] Added promotion features")
        display(self.df[[promo_col, "promo_streak"]].sample(10))

        # Visualization
        self.viz.plot_promotion_streak_distribution()

    # ---------------------------
    # Price Features + Viz
    # ---------------------------
    def add_price_features(self, price_col: str = "sell_price") -> None:
        """Create price-related features and visualize distributions"""
        if price_col not in self.df.columns:
            print(f"[WARN] price_col '{price_col}' not in DataFrame; skipping price features.")
            return
            
        # Feature creation
        self.df["price_item_mean"] = self.df.groupby("item_nbr")[price_col].transform("mean")
        self.df["price_rel_to_item"] = self.df[price_col] / (self.df["price_item_mean"].replace(0, np.nan))
        self.df["price_r7_std"] = self.df.groupby(["item_nbr", "store_nbr"])[price_col].transform(
            lambda s: s.rolling(7, min_periods=1).std()
        )
        
        print("[DEBUG] Added price features")
        display(self.df[[price_col, "price_rel_to_item", "price_r7_std"]].head(3))

        # Visualization
        self.viz.plot_price_relative_distribution()

    # ---------------------------
    # Holiday distance (FIXED)
    # ---------------------------
    def add_holiday_distance(
        self,
        holidays_df: pd.DataFrame,
        date_col: str = "date",
        holiday_date_col: str = "date"
    ) -> None:
        """Create holiday proximity features and visualize distributions (with detailed debug)."""

        print("🔧 Starting holiday distance feature generation...")
        print(f"   ➤ Using main date column: {date_col}")
        print(f"   ➤ Using holidays date column: {holiday_date_col}\n")

        # 1. Preparation
        print("1️⃣ Preparing data...")
        self._prepare(date_col)
        print("   ✔ self.df prepared and sorted by date.\n")

        holidays_df = holidays_df.copy()
        print(f"   • Holidays DataFrame shape: {holidays_df.shape}")

        # Ensure holiday dates are datetime
        print("   • Converting holiday dates to datetime...")
        holidays_df[holiday_date_col] = pd.to_datetime(holidays_df[holiday_date_col])
        holiday_dates = sorted(holidays_df[holiday_date_col].unique())
        print(f"   • Unique holiday dates found: {len(holiday_dates)}")
        if len(holiday_dates) > 0:
            print(f"     First 5 holiday dates: {holiday_dates[:5]}\n")
        else:
            print("     ⚠️ No holiday dates found.\n")

        # Array of holiday dates for efficient NumPy search
        hd = np.array(holiday_dates, dtype="datetime64[ns]")
        print(f"   • NumPy holiday array shape: {hd.shape}\n")

        # 2. Helper functions with robust type handling
        print("2️⃣ Defining helper functions days_until() and days_since()...\n")

        def days_until(dt):
            # Check if input is a valid datetime object (not NaT, not a number)
            if pd.isna(dt) or not isinstance(dt, (pd.Timestamp, np.datetime64)):
                return np.nan

            dt_np = np.datetime64(dt, 'ns')
            idx = np.searchsorted(hd, dt_np)

            if idx >= len(hd):
                return np.nan  # No future holiday found

            return (hd[idx] - dt_np).astype('timedelta64[D]').astype(int)

        def days_since(dt):
            if pd.isna(dt) or not isinstance(dt, (pd.Timestamp, np.datetime64)):
                return np.nan

            dt_np = np.datetime64(dt, 'ns')
            idx = np.searchsorted(hd, dt_np) - 1

            if idx < 0:
                return np.nan  # No past holiday found

            return (dt_np - hd[idx]).astype('timedelta64[D]').astype(int)

        # 3. Feature creation
        print("3️⃣ Creating holiday distance features on self.df...")
        print(f"   • self.df shape before: {self.df.shape}")
        print("   • Applying days_until()...")
        self.df["days_until_holiday"] = self.df[date_col].apply(days_until)
        print("   • Applying days_since()...")
        self.df["days_since_holiday"] = self.df[date_col].apply(days_since)
        print(f"   • self.df shape after: {self.df.shape}\n")

        # Quick sanity check
        print("4️⃣ Sample of computed holiday distance features (non-null rows):")
        sample = (
            self.df[["date", "days_until_holiday", "days_since_holiday"]]
            .dropna(how="all")
            .head(5)
        )
        if sample.empty:
            print("   ⚠️ No non-null holiday distance values found.")
        else:
            print(sample, "\n")

        print("[DEBUG] Added holiday distance features (days_until_holiday, days_since_holiday)\n")

        # 5. Visualization
        print("5️⃣ Plotting holiday distance distribution...")
        self.viz.plot_holiday_distance_distribution()
        print("🎉 Holiday distance feature generation and visualization complete.\n")

    # ---------------------------
    # Store & Item Aggregates + Viz
    # ---------------------------
    def add_store_item_aggregates(self, sales_col: str = "unit_sales") -> None:
        """Create store and item level aggregates and visualize distributions (with detailed debug)."""

        print("🔧 Starting store/item aggregate feature generation...")
        print(f"   ➤ Using sales column: {sales_col}\n")

        # 1️⃣ Basic info
        print("1️⃣ DataFrame overview before aggregation:")
        print(f"   • Shape: {self.df.shape}")
        print(f"   • Columns: {list(self.df.columns)}\n")

        # 2️⃣ Store-level average sales
        print("2️⃣ Computing store-level average sales...")
        self.df["store_avg_sales"] = (
            self.df.groupby("store_nbr")[sales_col].transform("mean")
        )
        print("   ✔ store_avg_sales created.")
        print(f"   • Non-null values: {self.df['store_avg_sales'].notna().sum()}\n")

        # 3️⃣ Item-level average sales
        print("3️⃣ Computing item-level average sales...")
        self.df["item_avg_sales"] = (
            self.df.groupby("item_nbr")[sales_col].transform("mean")
        )
        print("   ✔ item_avg_sales created.")
        print(f"   • Non-null values: {self.df['item_avg_sales'].notna().sum()}\n")

        # 4️⃣ Item popularity rank (based on mean sales)
        print("4️⃣ Computing item popularity rank (dense rank on item_avg_sales)...")
        item_mean = self.df.groupby("item_nbr")[sales_col].transform("mean")
        self.df["item_popularity_rank"] = item_mean.rank(method="dense", ascending=False)
        print("   ✔ item_popularity_rank created.")
        print(f"   • Rank range: {self.df['item_popularity_rank'].min()} → {self.df['item_popularity_rank'].max()}\n")

        # 5️⃣ Store-item median sales
        print("5️⃣ Computing store-item median sales...")
        self.df["store_item_median"] = (
            self.df.groupby(["store_nbr", "item_nbr"])[sales_col].transform("median")
        )
        print("   ✔ store_item_median created.")
        print(f"   • Non-null values: {self.df['store_item_median'].notna().sum()}\n")

        # 6️⃣ Sample preview
        print("6️⃣ Sample of aggregate features:")
        sample_cols = [
            "store_nbr", "item_nbr",
            "store_avg_sales", "item_avg_sales",
            "item_popularity_rank", "store_item_median"
        ]
        display(self.df[sample_cols].head(5))
        print("\n[DEBUG] Added store/item aggregate features.\n")

        # 7️⃣ Visualization
        print("7️⃣ Plotting store average sales distribution...")
        self.viz.plot_store_avg_sales()
        print("🎉 Store/item aggregate feature generation and visualization complete.\n")

    # ---------------------------
    # Feature Selection Helpers
    # ---------------------------
    def remove_low_variance(self, threshold: float = 1e-3, inplace: bool = True) -> pd.DataFrame:
        """Remove low variance features and return statistics"""
        numeric = self.df.select_dtypes(include=[np.number])
        selector = VarianceThreshold(threshold)
        selector.fit(numeric.fillna(0))
        cols_to_keep = numeric.columns[selector.get_support(indices=True)].tolist()
        removed = [c for c in numeric.columns if c not in cols_to_keep]
        
        if inplace:
            self.df = pd.concat([self.df.drop(columns=numeric.columns, errors='ignore'), 
                                numeric[cols_to_keep]], axis=1)
            
        print(f"[DEBUG] Removed {len(removed)} low-variance features")
        
        return pd.DataFrame({
            "kept_count": [len(cols_to_keep)], 
            "removed_count": [len(removed)]
        })

    def calculate_vif(self, features: Optional[Iterable[str]] = None, top_n: int = 10) -> pd.DataFrame:
        """Calculate Variance Inflation Factor for features"""
        numeric = self.df.select_dtypes(include=[np.number]).copy()
        if features:
            numeric = numeric[list(features)]
            
        numeric = numeric.dropna(axis=1, how="all").fillna(0)
        cols = numeric.columns.tolist()
        vif = []
        
        for i, col in enumerate(cols):
            try:
                vif_val = variance_inflation_factor(numeric.values, i)
            except Exception:
                vif_val = np.nan
            vif.append({"feature": col, "vif": vif_val})
            
        vif_df = pd.DataFrame(vif).sort_values("vif", ascending=False).reset_index(drop=True)
        display(vif_df.head(top_n))
        
        return vif_df

    # ---------------------------
    # Export Final Dataset
    # ---------------------------
    def save_final_dataset(self, filename: Optional[str] = None) -> str:
        """Save the engineered dataset to disk"""
        if filename is None:
            filename = f"final_dataset_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv"
        save_path = os.path.join(self.processed_path, filename)
        self.df.to_csv(save_path, index=False)
        print(f"💾 Final dataset saved to {save_path}")
        return save_path

    # ---------------------------
    # Convenience: Run Full Feature Pipeline
    # ---------------------------
    def create_all_features(self,
                            date_col: str = "date",
                            sales_col: str = "unit_sales",
                            price_col: Optional[str] = "sell_price",
                            holidays_df: Optional[pd.DataFrame] = None,
                            price_ops: bool = True) -> str:
        """
        Execute complete feature engineering pipeline with integrated visualization
        
        Parameters:
        -----------
        date_col : str
            Name of the date column
        sales_col : str
            Name of the sales/target column
        price_col : Optional[str]
            Name of the price column
        holidays_df : Optional[pd.DataFrame]
            DataFrame containing holiday dates
        price_ops : bool
            Whether to include price features
            
        Returns:
        --------
        str : Path to the saved dataset
        """
        print("🚀 Running full feature engineering pipeline...")
        
        # Feature Creation with Integrated Visualization
        self.add_date_features(date_col=date_col)
        self.add_target_transform(sales_col)
        self.add_advanced_lags(sales_col=sales_col, lags=(7, 14, 28, 365))
        self.add_rolling_features(sales_col=sales_col, windows=(7, 28, 365), stats=("mean", "std"))
        self.add_promotion_features(promo_col="onpromotion", promo_lags=(1, 7))
        
        if price_ops and price_col is not None:
            self.add_price_features(price_col=price_col)
        
        if holidays_df is not None:
            self.add_holiday_distance(holidays_df, date_col=date_col, holiday_date_col="date")
        
        self.add_store_item_aggregates(sales_col=sales_col)

        # Feature Selection with Visualization
        print("\n🔍 Running feature selection...")
        low_var_results = self.remove_low_variance(threshold=1e-5)
        self.viz.plot_low_variance_selection(
            removed_count=low_var_results["removed_count"].iloc[0], 
            kept_count=low_var_results["kept_count"].iloc[0]
        )
        
        try:
            vif_df = self.calculate_vif(top_n=20)
            self.viz.plot_vif_top_features(vif_df, top_n=20)
        except Exception as e:
            print(f"[WARN] VIF calculation failed: {e}")

        # Export Results
        saved_path = self.save_final_dataset()
        print("✅ Feature engineering pipeline complete.")
        return saved_path