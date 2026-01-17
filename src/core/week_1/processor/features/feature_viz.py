
import os
import pandas as pd  # type: ignore  # noqa: F401
import numpy as np  # type: ignore  # noqa: F401
import seaborn as sns # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from IPython.display import display  # type: ignore  # noqa: F401

from src.utils import get_path
from typing import Iterable




class FeatureViz:
    """
    Handles all visualization related to the FeatureEngineering process.
    Saves plots to the designated features_viz path.
    Uses consistent figsize (12, 8) and dark color palette.
    """
    
    # Class-level constants for consistency
    FIGSIZE = (12, 8)
    DARK_PALETTE = {
        'primary': '#1a1a2e',      # Very dark blue
        'secondary': '#0f3460',    # Dark blue
        'accent': '#16213e',       # Navy
        'line1': '#2c3e50',        # Dark slate
        'line2': '#34495e',        # Darker gray-blue
        'highlight': '#1c4966',    # Deep teal
        'contrast': '#0d1b2a',     # Almost black blue
    }
    
    def __init__(self, df: pd.DataFrame, week: int = 1):
        self.df = df.copy()
        self.week = week
        self.viz_path = get_path("features_viz", week=self.week)
        os.makedirs(self.viz_path, exist_ok=True)
        print(f"[DEBUG] Initialized FeatureViz for week {self.week}")
    
    def _save_plot(self, name: str):
        """Utility to save and display plot."""
        path = os.path.join(self.viz_path, name)
        plt.savefig(path, dpi=300, bbox_inches="tight")
        print(f"💾 Plot saved → {path}")

    # --- Date Features Visualizations ---

    def plot_daily_coverage(self, date_col: str = "date", sales_col: str = "unit_sales"):
        """Plot daily record count to show coverage over time."""
        plt.figure(figsize=self.FIGSIZE)
        daily_counts = self.df.groupby(date_col)[sales_col].size()
        sns.lineplot(x=daily_counts.index, y=daily_counts.values, color=self.DARK_PALETTE['secondary'])
        plt.title("Daily Record Coverage", fontsize=16, fontweight="bold")
        plt.xlabel("Date")
        plt.ylabel("Row count")
        plt.grid(axis='y', linestyle='--', alpha=0.3)
        self._save_plot("date_features_daily_coverage.png")
        plt.show()

    def plot_monthly_total_sales(self, date_col: str = "date", sales_col: str = "unit_sales"):
        """Plot monthly total unit sales trend."""
        df_copy = self.df.copy()
        df_copy["year"] = df_copy[date_col].dt.year
        df_copy["month"] = df_copy[date_col].dt.month
        
        plt.figure(figsize=self.FIGSIZE)
        monthly_sales = df_copy.groupby(["year", "month"])[sales_col].sum().reset_index()
        monthly_sales["year_month"] = pd.to_datetime(
            monthly_sales["year"].astype(str) + "-" + monthly_sales["month"].astype(str) + "-01"
        )
        sns.lineplot(x="year_month", y=sales_col, data=monthly_sales, color=self.DARK_PALETTE['primary'], linewidth=2.5)
        plt.title("Monthly Total Unit Sales", fontsize=16, fontweight="bold")
        plt.xlabel("Month")
        plt.ylabel("Sales")
        plt.grid(axis='y', linestyle='--', alpha=0.3)
        self._save_plot("date_features_monthly_sales.png")
        plt.show()

    def plot_monthly_sales_heatmap(self, year_col="year", month_col="month", sales_col="unit_sales"):
        """Plot monthly sales trends over years as a heatmap."""
        df_copy = self.df.copy()
        df_copy["year"] = df_copy["date"].dt.year
        df_copy["month"] = df_copy["date"].dt.month
        sales_by_month = df_copy.groupby(["year", "month"])[sales_col].sum().unstack()
        
        plt.figure(figsize=self.FIGSIZE)
        sns.heatmap(sales_by_month, cmap="Blues", linewidths=0.5, linecolor="black", 
                    cbar_kws={"label": "Sales Volume"})
        plt.title("Monthly Sales Trends Over Years", fontsize=16, fontweight="bold")
        plt.tight_layout()
        self._save_plot(f"{sales_col}_monthly_sales_heatmap.png")
        plt.show()

    def plot_sales_by_day_of_week(self, sales_col="unit_sales"):
        """Plot average sales by day of week."""
        df_copy = self.df.copy()
        df_copy["day_of_week"] = df_copy["date"].dt.dayofweek
        sales_by_dow = df_copy.groupby("day_of_week")[sales_col].mean()
        
        plt.figure(figsize=self.FIGSIZE)
        colors = [self.DARK_PALETTE['secondary'], self.DARK_PALETTE['primary'], 
                  self.DARK_PALETTE['accent'], self.DARK_PALETTE['line1'],
                  self.DARK_PALETTE['line2'], self.DARK_PALETTE['highlight'],
                  self.DARK_PALETTE['contrast']]
        sns.barplot(x=sales_by_dow.index, y=sales_by_dow.values, palette=colors)
        plt.title("Average Sales by Day of Week", fontsize=16, fontweight="bold")
        plt.xlabel("Day of Week")
        plt.ylabel("Average Sales")
        plt.grid(axis='y', linestyle='--', alpha=0.3)
        plt.tight_layout()
        self._save_plot(f"{sales_col}_sales_by_day_of_week.png")
        plt.show()
        
    def plot_monthly_boxplot(self, sales_col="unit_sales"):
        """Plot distribution of sales by month."""
        df_copy = self.df.copy()
        df_copy["month"] = df_copy["date"].dt.month
        
        plt.figure(figsize=self.FIGSIZE)
        sns.boxplot(x=df_copy["month"], y=df_copy[sales_col], 
                    color=self.DARK_PALETTE['secondary'], 
                    boxprops=dict(facecolor=self.DARK_PALETTE['accent']))
        plt.title("Distribution of Sales by Month", fontsize=16, fontweight="bold")
        plt.xlabel("Month")
        plt.ylabel("Unit Sales")
        plt.grid(axis='y', linestyle='--', alpha=0.3)
        plt.tight_layout()
        self._save_plot("monthly_boxplot.png")
        plt.show()
        
    def plot_sales_by_store(self, date_col="date", sales_col="unit_sales", top_n=5):
        """Plot sales trends for top N stores."""
        top_stores = self.df.groupby("store_nbr")[sales_col].sum().nlargest(top_n).index
        subset = self.df[self.df["store_nbr"].isin(top_stores)]
        
        plt.figure(figsize=self.FIGSIZE)
        dark_colors = [self.DARK_PALETTE['primary'], self.DARK_PALETTE['secondary'],
                       self.DARK_PALETTE['accent'], self.DARK_PALETTE['line1'],
                       self.DARK_PALETTE['line2']]
        sns.lineplot(data=subset, x=date_col, y=sales_col, hue="store_nbr", 
                     palette=dark_colors[:top_n], linewidth=2)
        plt.title(f"Sales Trends for Top {top_n} Stores", fontsize=16, fontweight="bold")
        plt.grid(axis='y', linestyle='--', alpha=0.3)
        plt.tight_layout()
        self._save_plot(f"{sales_col}_sales_by_store.png")
        plt.show()
        
    def plot_year_over_year(self, date_col="date", sales_col="unit_sales"):
        """Plot year-over-year sales comparison."""
        df_copy = self.df.copy()
        df_copy["day_of_year"] = df_copy[date_col].dt.dayofyear
        df_copy["year"] = df_copy[date_col].dt.year
        
        plt.figure(figsize=self.FIGSIZE)
        years = df_copy["year"].unique()
        dark_palette = [self.DARK_PALETTE['primary'], self.DARK_PALETTE['secondary'],
                        self.DARK_PALETTE['accent'], self.DARK_PALETTE['line1'],
                        self.DARK_PALETTE['line2'], self.DARK_PALETTE['highlight']]
        sns.lineplot(data=df_copy, x="day_of_year", y=sales_col, hue="year", 
                     palette=dark_palette[:len(years)], linewidth=2)
        plt.title("Year-over-Year Sales Comparison", fontsize=16, fontweight="bold")
        plt.xlabel("Day of Year")
        plt.ylabel(f"{sales_col.replace('_', ' ').title()}")
        plt.grid(axis='y', linestyle='--', alpha=0.3)
        plt.tight_layout()
        self._save_plot(f"{sales_col}_year_over_year.png")
        plt.show()

    # --- Target Transforms Visualizations ---

    def plot_target_distribution(self, sales_col: str = "unit_sales"):
        """Plot distribution of original vs log-transformed target."""
        log_col = f"{sales_col}_log"
        if log_col not in self.df.columns:
             print(f"[WARN] {log_col} not found for plotting.")
             return
             
        plt.figure(figsize=self.FIGSIZE)
        sns.histplot(self.df[sales_col], kde=True, color=self.DARK_PALETTE['secondary'], 
                     label="Original", log_scale=(False, False), alpha=0.7)
        sns.histplot(self.df[log_col], kde=True, color=self.DARK_PALETTE['primary'], 
                     label="Log(1+x)", log_scale=(False, False), alpha=0.7)
        plt.legend()
        plt.title("Target distribution: Original vs Log(1+x)", fontsize=16, fontweight="bold")
        plt.grid(axis='y', linestyle='--', alpha=0.3)
        self._save_plot("target_transform_distribution.png")
        plt.show()

    def plot_pct_change_distribution(self, sales_col: str = "unit_sales", window: int = 7):
        """Plot distribution of N-day percent change of target."""
        pct_col = f"{sales_col}_pct_change_{window}"
        if pct_col not in self.df.columns:
             print(f"[WARN] {pct_col} not found for plotting.")
             return
             
        plt.figure(figsize=self.FIGSIZE)
        sns.histplot(self.df[pct_col].dropna(), bins=100, color=self.DARK_PALETTE['accent'])
        plt.title(f"{window}-day Percent Change Distribution", fontsize=16, fontweight="bold")
        plt.grid(axis='y', linestyle='--', alpha=0.3)
        self._save_plot(f"target_pct_change_{window}.png")
        plt.show()

    # --- Lag Features Visualizations ---

    def plot_lags_overlay(self, sales_col: str = "unit_sales", lags: Iterable[int] = (7, 28, 365)):
        """Plot actual sales overlaid with selected lag features for a sample store/item."""
        valid_lags = [l for l in lags if f"{sales_col}_lag_{l}" in self.df.columns]  # noqa: E741
        if not valid_lags:
            print("[WARN] No valid lag features found for plotting.")
            return

        plt.figure(figsize=self.FIGSIZE)
        sample_key = self.df.groupby(["store_nbr", "item_nbr"]).size().idxmax()
        subset = self.df[(self.df["store_nbr"] == sample_key[0]) & (self.df["item_nbr"] == sample_key[1])]
        
        if subset.empty:
            print("[WARN] Sample store/item subset is empty.")
            return

        sns.lineplot(data=subset, x="date", y=sales_col, label="Actual", 
                     color="black", linewidth=2.5)
        colors = [self.DARK_PALETTE['secondary'], self.DARK_PALETTE['accent'], 
                  self.DARK_PALETTE['line1']]
        for i, lag in enumerate(valid_lags):
            sns.lineplot(data=subset, x="date", y=f"{sales_col}_lag_{lag}", 
                        label=f"Lag {lag}", alpha=0.7, color=colors[i % len(colors)],
                        linewidth=2)
            
        plt.title(f"Lag Overlays for store={sample_key[0]} item={sample_key[1]}", 
                  fontsize=16, fontweight="bold")
        plt.grid(axis='y', linestyle='--', alpha=0.3)
        self._save_plot("lags_overlay_sample_store_item.png")
        plt.show()

    # --- Rolling Features Visualizations ---

    def plot_rolling_mean_std(self, sales_col: str = "unit_sales", window: int = 7):
        """Plot rolling mean vs rolling standard deviation for a sample store/item."""
        mean_col = f"{sales_col}_r{window}_mean"
        std_col = f"{sales_col}_r{window}_std"
        if mean_col not in self.df.columns or std_col not in self.df.columns:
            print(f"[WARN] Rolling features ({mean_col} or {std_col}) not found for plotting.")
            return

        plt.figure(figsize=self.FIGSIZE)
        key = self.df.groupby(["store_nbr", "item_nbr"]).size().idxmax()
        sub = self.df[(self.df["store_nbr"] == key[0]) & (self.df["item_nbr"] == key[1])]

        if sub.empty:
            print("[WARN] Sample store/item subset is empty.")
            return
            
        sns.lineplot(sub, x="date", y=mean_col, label=f"r{window}_mean", 
                     color=self.DARK_PALETTE['secondary'], linewidth=2.5)
        sns.lineplot(sub, x="date", y=std_col, label=f"r{window}_std", 
                     color=self.DARK_PALETTE['accent'], linewidth=2.5)
        plt.title(f"Rolling Mean vs Std (store={key[0]}, item={key[1]}, Window={window})", 
                  fontsize=16, fontweight="bold")
        plt.grid(axis='y', linestyle='--', alpha=0.3)
        self._save_plot("rolling_mean_std_sample.png")
        plt.show()
        
    
    def _select_valid_store_item(self, metric, windows, method, group_cols):
        """
        Select a store-item pair that actually has valid rolling features,
        with detailed step-by-step debug output.
        """

        print("🔍 Selecting valid store-item for rolling smoothing plot...")
        print(f"   ➤ Metric: {metric}")
        print(f"   ➤ Windows: {windows}")
        print(f"   ➤ Method: {method}")
        print(f"   ➤ Group columns: {group_cols}\n")

        # 1) Build list of required rolling columns
        required_cols = [f"{metric}_r{w}_{method}" for w in windows]
        print("1️⃣ Required rolling columns:")
        for c in required_cols:
            print(f"   • {c} (exists: {c in self.df.columns})")
        print()

        # 2) Check how many non-null values each required column has
        print("2️⃣ Non-null counts per required column (global):")
        for c in required_cols:
            if c in self.df.columns:
                print(f"   • {c}: {self.df[c].notna().sum()} non-null")
            else:
                print(f"   • {c}: MISSING COLUMN")
        print()

        # 3) Drop rows where ALL rolling columns are NaN
        print("3️⃣ Filtering rows where at least one rolling column is non-null...")
        df_valid = self.df.dropna(subset=required_cols, how="all")
        print(f"   • Original rows: {len(self.df)}")
        print(f"   • Rows with at least one valid rolling value: {len(df_valid)}\n")

        if df_valid.empty:
            print("❌ No rows with valid rolling features found.")
            raise ValueError("No store-item pair has valid rolling features.")

        # 4) Group by store-item and count valid rows
        print("4️⃣ Grouping valid rows by store-item and counting...")
        group_sizes = df_valid.groupby(group_cols).size().sort_values(ascending=False)
        print("   • Top 5 store-item pairs by valid rows:")
        print(group_sizes.head(5), "\n")

        # 5) Pick the store-item with the most valid rows
        key = group_sizes.idxmax()
        print(f"5️⃣ Selected store-item key: {group_cols[0]}={key[0]}, {group_cols[1]}={key[1]}")
        print(f"   • Valid rows for this pair: {group_sizes.max()}\n")

        return key



    def plot_rolling_smoothing_trends(
        self,
        metric="unit_sales",
        windows=[3, 7, 14, 30, 365],
        method="mean",
        group_cols=["store_nbr", "item_nbr"]
    ):
        """
        Plot all rolling smoothing windows for a representative store-item pair
        in one professional Plotly diagram.
        """

        # Pick the store-item with most records
        key = self._select_valid_store_item(metric, windows, method, group_cols)
        sub = self.df[
            (self.df[group_cols[0]] == key[0]) &
            (self.df[group_cols[1]] == key[1])
        ]

        if sub.empty:
            print("[WARN] No data for selected store-item.")
            return

        fig = go.Figure()

        # Add each rolling window
        for w in windows:
            col = f"{metric}_r{w}_{method}"
            if col not in sub.columns or sub[col].notna().sum() == 0:
                print(f"[WARN] Missing column: {col}")
                continue

            fig.add_trace(
                go.Scatter(
                    x=sub["date"],
                    y=sub[col],
                    mode="lines",
                    name=f"{w}-day {method}",
                    line=dict(width=2)
                )
            )

        # Add actual sales
        fig.add_trace(
            go.Scatter(
                x=sub["date"],
                y=sub[metric],
                mode="lines",
                name="Actual",
                line=dict(width=3, color="white")
            )
        )

        fig.update_layout(
            title=f"Rolling Smoothing Trends for store={key[0]}, item={key[1]}",
            template="plotly_dark",
            height=900,
            width=1200,
            margin=dict(l=40, r=40, t=80, b=40)
        )

        # Save image
        save_path = os.path.join(self.viz_path, f"rolling_smoothing_{method}trends.png")
        fig.write_image(save_path, scale=2)
        print(f"💾 Plot saved → {save_path}")

        fig.show()


    # --- Promotion Features Visualizations ---

    def plot_promotion_streak_distribution(self, promo_streak_col: str = "promo_streak"):
        """Plot distribution of promotion streak length."""
        if promo_streak_col not in self.df.columns:
            print(f"[WARN] {promo_streak_col} not found for plotting.")
            return

        plt.figure(figsize=self.FIGSIZE)
        sns.histplot(self.df[promo_streak_col].clip(upper=60).dropna(), bins=60, 
                     color=self.DARK_PALETTE['primary'])
        plt.title("Promotion Streak Length Distribution (capped at 60)", 
                  fontsize=16, fontweight="bold")
        plt.grid(axis='y', linestyle='--', alpha=0.3)
        self._save_plot("promotion_streak_distribution.png")
        plt.show()

    # --- Price Features Visualizations ---

    def plot_price_relative_distribution(self, rel_price_col: str = "price_rel_to_item"):
        """Plot distribution of price relative to item mean."""
        if rel_price_col not in self.df.columns:
            print(f"[WARN] {rel_price_col} not found for plotting.")
            return
            
        plt.figure(figsize=self.FIGSIZE)
        sns.histplot(
            self.df[rel_price_col].replace([np.inf, -np.inf], np.nan).dropna(), 
            bins=100, 
            color=self.DARK_PALETTE['highlight']
        )
        plt.title("Relative Price to Item Mean Distribution", fontsize=16, fontweight="bold")
        plt.grid(axis='y', linestyle='--', alpha=0.3)
        self._save_plot("price_relative_distribution.png")
        plt.show()

    # --- Holiday Distance Visualizations ---

    def plot_holiday_distance_distribution(self): 
        """Plot distribution of days until and since a holiday."""
        until_col = "days_until_holiday"
        since_col = "days_since_holiday"
        
        if until_col not in self.df.columns or since_col not in self.df.columns:
             print(f"[WARN] Holiday distance features ({until_col} or {since_col}) not found for plotting.")
             return
             
        plt.figure(figsize=self.FIGSIZE)
        
        sns.histplot(
            self.df[until_col].dropna().clip(upper=100), 
            bins=50, 
            color=self.DARK_PALETTE['secondary'], 
            label="Days Until Holiday (capped at 100)", 
            kde=True,
            alpha=0.7
        )
        
        sns.histplot(
            self.df[since_col].dropna().clip(upper=100), 
            bins=50, 
            color=self.DARK_PALETTE['accent'], 
            label="Days Since Holiday (capped at 100)", 
            kde=True,
            alpha=0.6
        )
        
        plt.legend()
        plt.title("Distance to Nearest Holiday (Days)", fontsize=16, fontweight="bold")
        plt.xlabel("Days")
        plt.ylabel("Frequency")
        plt.grid(axis='y', linestyle='--', alpha=0.3)
        plt.tight_layout()
        
        self._save_plot("holiday_distance_distribution.png")
        plt.show()
    
    
    def plot_store_avg_sales(self, sales_col: str = "store_avg_sales", top_n: int = 20) -> None:
        """
        Plot distribution of average sales per store.
        Uses the precomputed `store_avg_sales` column and shows the top N stores.
        """

        print(" Plotting store average sales distribution...")
        if sales_col not in self.df.columns:
            print(f"[WARN] {sales_col} not found in DataFrame. Did you run add_store_item_aggregates?")
            return

        # Aggregate once per store
        print("   ➤ Aggregating average sales per store...")
        store_stats = (
            self.df.groupby("store_nbr")[sales_col]
            .mean()
            .sort_values(ascending=False)
            .reset_index()
        )
        print(f"   • Total stores: {len(store_stats)}")
        print(f"   • Showing top {min(top_n, len(store_stats))} stores.\n")

        top_stores = store_stats.head(top_n)

        plt.figure(figsize=self.FIGSIZE)
        sns.barplot(
            data=top_stores,
            x="store_nbr",
            y=sales_col,
            palette=[self.DARK_PALETTE["secondary"]] * len(top_stores),
        )
        plt.title(f"Top {top_n} Stores by Average Unit Sales", fontsize=16, fontweight="bold")
        plt.xlabel("Store Number")
        plt.ylabel("Average Unit Sales")
        plt.grid(axis="y", linestyle="--", alpha=0.3)
        plt.tight_layout()

        self._save_plot("store_avg_sales_distribution.png")
        plt.show()

        print("✅ Store average sales plot generated and saved.\n")

    
    

    def plot_calendar_feature_trends(self, date_col="date", sales_col="unit_sales"):
        """
        Create a single  Plotly dashboard showing trends of unit_sales
        across all calendar-based features, and save it to the features_viz folder.
        """

        df = self.df.copy()

        # Ensure date features exist
        # Add date features only if they are not already present
        if "year" not in df.columns:
            df["year"] = df[date_col].dt.year

        if "month" not in df.columns:
            df["month"] = df[date_col].dt.month

        if "day_of_week" not in df.columns:
            df["day_of_week"] = df[date_col].dt.dayofweek

        if "week_of_year" not in df.columns:
            df["week_of_year"] = df[date_col].dt.isocalendar().week.astype(int)

        if "quarter" not in df.columns:
            df["quarter"] = df[date_col].dt.quarter

        if "is_weekend" not in df.columns:
            df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

        if "day_of_year" not in df.columns:
            df["day_of_year"] = df[date_col].dt.dayofyear


        # Aggregations
        agg_year = df.groupby("year")[sales_col].mean()
        agg_month = df.groupby("month")[sales_col].mean()
        agg_dow = df.groupby("day_of_week")[sales_col].mean()
        agg_woy = df.groupby("week_of_year")[sales_col].mean()
        agg_quarter = df.groupby("quarter")[sales_col].mean()
        agg_weekend = df.groupby("is_weekend")[sales_col].mean()
        agg_doy = df.groupby("day_of_year")[sales_col].mean()

        # Create subplot layout
        fig = make_subplots(
            rows=4, cols=2,
            subplot_titles=[
                "Avg Sales by Year",
                "Avg Sales by Month",
                "Avg Sales by Day of Week",
                "Avg Sales by Week of Year",
                "Avg Sales by Quarter",
                "Avg Sales: Weekend vs Weekday",
                "Avg Sales by Day of Year",
                ""
            ],
            vertical_spacing=0.12,
            horizontal_spacing=0.1
        )

        # Row 1
        fig.add_trace(go.Scatter(x=agg_year.index, y=agg_year.values, mode="lines+markers",
                                name="Year"), row=1, col=1)

        fig.add_trace(go.Scatter(x=agg_month.index, y=agg_month.values, mode="lines+markers",
                                name="Month"), row=1, col=2)

        # Row 2
        fig.add_trace(go.Bar(x=agg_dow.index, y=agg_dow.values, name="Day of Week"),
                    row=2, col=1)

        fig.add_trace(go.Scatter(x=agg_woy.index, y=agg_woy.values, mode="lines",
                                name="Week of Year"), row=2, col=2)

        # Row 3
        fig.add_trace(go.Bar(x=agg_quarter.index, y=agg_quarter.values, name="Quarter"),
                    row=3, col=1)

        fig.add_trace(go.Bar(x=["Weekday", "Weekend"], y=agg_weekend.values,
                            name="Weekend"), row=3, col=2)

        # Row 4
        fig.add_trace(go.Scatter(x=agg_doy.index, y=agg_doy.values, mode="lines",
                                name="Day of Year"), row=4, col=1)

        # Layout styling
        fig.update_layout(
            height=1600,
            width=1200,
            title_text="Calendar Feature Trends for Unit Sales",
            showlegend=False,
            template="plotly_dark",
            title_font=dict(size=22),
            margin=dict(l=40, r=40, t=80, b=40)
        )

        # --- SAVE FIGURE ---
        save_path = os.path.join(self.viz_path, "calendar_feature_trends.png")
        fig.write_image(save_path, scale=2)
        print(f"💾 Plot saved → {save_path}")

        fig.show()
