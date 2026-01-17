# src/core/week_1/processor/eda/visualization.py

import os
import pandas as pd # type: ignore
import matplotlib.pyplot as plt # type: ignore
import seaborn as sns # type: ignore
from statsmodels.graphics.tsaplots import plot_acf # type: ignore
from src.utils import get_path


class Visualization:
    """
    Optimized visualization class for large-scale Favorita Grocery Sales data.
    
    Key Optimizations:
      • Pre-aggregate data in __init__ to avoid repeated groupby operations
      • Cache expensive computations
      • Use efficient pandas operations
      • Downsample for visualizations that don't need full granularity
    """

    def __init__(self, df, pre_aggregate=True):
        """
        Args:
            df (pd.DataFrame): Raw sales data
            pre_aggregate (bool): If True, pre-compute aggregations for faster plotting
        """
        self.df_raw = df  # Keep reference to raw data
        self.eda_path = get_path("eda", week=1)
        os.makedirs(self.eda_path, exist_ok=True)
        
        print(f"[DEBUG] Initialized Visualization with DataFrame of shape {df.shape}")
        
        # Pre-aggregate common queries for fast plotting
        if pre_aggregate:
            print("[DEBUG] Pre-aggregating data for faster visualization...")
            self._precompute_aggregations()
        else:
            self.df = df.copy()
    
    def _precompute_aggregations(self):
        """Pre-compute expensive aggregations once during initialization."""
        df = self.df_raw.copy()
        
        # Ensure date is datetime
        df['date'] = pd.to_datetime(df['date'])
        
        # 1. Daily total sales (used by multiple plots)
        print("  • Computing daily total sales...")
        self.daily_sales = df.groupby('date')['unit_sales'].sum().sort_index()
        
        # 2. Store-level daily sales (for top stores)
        print("  • Computing store-level aggregations...")
        store_totals = df.groupby('store_nbr')['unit_sales'].sum()
        self.top_stores = store_totals.nlargest(5).index.tolist()
        
        # Pre-aggregate only top stores
        self.store_daily_sales = df[df['store_nbr'].isin(self.top_stores)].groupby(
            ['date', 'store_nbr']
        )['unit_sales'].sum().reset_index()
        
        # 3. Keep full dataframe but with reduced memory
        # Only keep columns needed for visualization
        self.df = df[['date', 'unit_sales', 'store_nbr']].copy()
        
        print(f"[DEBUG] Pre-aggregation complete. Memory reduced significantly.")  # noqa: F541

    def plot_total_sales_over_time(self, date_col="date", sales_col="unit_sales"):
        """Optimized: Uses pre-aggregated daily sales."""
        print(f"[DEBUG] Plotting total sales over time...")  # noqa: F541
        
        plt.figure(figsize=(12, 6))
        plt.plot(self.daily_sales.index, self.daily_sales.values)
        plt.title(f"Total {sales_col.replace('_', ' ').title()} Over Time", 
                  fontsize=20, fontweight="bold")
        plt.xlabel("Date")
        plt.ylabel(f"{sales_col.replace('_', ' ').title()}")
        plt.tight_layout()
        
        save_path = os.path.join(self.eda_path, f"{sales_col}_total_sales_over_time.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"💾 Total sales over time plot saved to {save_path}")
        plt.show()

    def plot_monthly_sales_heatmap(self, year_col="year", month_col="month", sales_col="unit_sales"):
        """Optimized: Aggregate on pre-computed daily sales."""
        print(f"[DEBUG] Plotting monthly sales heatmap...")  # noqa: F541
        
        # Use daily aggregated data
        df_monthly = self.daily_sales.to_frame()
        df_monthly['year'] = df_monthly.index.year
        df_monthly['month'] = df_monthly.index.month
        
        sales_by_month = df_monthly.groupby(['year', 'month'])['unit_sales'].sum().unstack()
        
        plt.figure(figsize=(8, 5))
        sns.heatmap(sales_by_month, cmap="coolwarm", linewidths=0.5, 
                    linecolor="white", cbar_kws={"label": "Sales Volume"})
        plt.title("Monthly Sales Trends Over Years", fontsize=22, fontweight="bold")
        plt.tight_layout()
        
        save_path = os.path.join(self.eda_path, f"{sales_col}_monthly_sales_heatmap.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"💾 Monthly sales heatmap saved to {save_path}")
        plt.show()

    def plot_sales_by_day_of_week(self, sales_col="unit_sales"):
        """Optimized: Uses daily aggregated data."""
        print(f"[DEBUG] Plotting sales by day of week...")  # noqa: F541
        
        # Use daily aggregated data
        df_daily = self.daily_sales.to_frame()
        df_daily['day_of_week'] = df_daily.index.dayofweek
        
        sales_by_dow = df_daily.groupby('day_of_week')['unit_sales'].mean()
        
        plt.figure(figsize=(8, 5))
        sns.barplot(x=sales_by_dow.index, y=sales_by_dow.values, palette="viridis")
        plt.title("Average Sales by Day of Week", fontsize=20, fontweight="bold")
        plt.xticks(range(7), ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
        plt.tight_layout()
        
        save_path = os.path.join(self.eda_path, f"{sales_col}_sales_by_day_of_week.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"💾 Sales by day-of-week plot saved to {save_path}")
        plt.show()

    def plot_monthly_boxplot(self, sales_col="unit_sales"):
        """Optimized: Uses daily aggregated data."""
        print(f"[DEBUG] Plotting monthly boxplot...")  # noqa: F541
        
        # Use daily aggregated data
        df_daily = self.daily_sales.to_frame()
        df_daily['month'] = df_daily.index.month
        
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=df_daily, x='month', y='unit_sales', palette="Set3")
        plt.title("Distribution of Daily Sales by Month")
        plt.xlabel("Month")
        plt.ylabel("Daily Unit Sales")
        plt.tight_layout()
        
        save_path = os.path.join(self.eda_path, "monthly_boxplot.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"💾 Monthly boxplot saved to {save_path}")
        plt.show()

    def plot_sales_by_store(self, date_col="date", sales_col="unit_sales", top_n=5):
        """Optimized: Uses pre-aggregated store data."""
        print(f"[DEBUG] Plotting sales by store (top {top_n})...")
        
        plt.figure(figsize=(12, 6))
        
        # Use pre-aggregated store data
        for store in self.top_stores[:top_n]:
            store_data = self.store_daily_sales[self.store_daily_sales['store_nbr'] == store]
            plt.plot(store_data['date'], store_data['unit_sales'], 
                    label=f'Store {store}', alpha=0.7)
        
        plt.title(f"Sales Trends for Top {top_n} Stores")
        plt.xlabel("Date")
        plt.ylabel("Unit Sales")
        plt.legend()
        plt.tight_layout()
        
        save_path = os.path.join(self.eda_path, f"{sales_col}_sales_by_store.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"💾 Sales by store plot saved to {save_path}")
        plt.show()

    def plot_year_over_year(self, date_col="date", sales_col="unit_sales"):
        """Optimized: Uses daily aggregated data."""
        print(f"[DEBUG] Plotting year-over-year comparison...")  # noqa: F541
        
        # Use daily aggregated data
        df_daily = self.daily_sales.to_frame()
        df_daily['year'] = df_daily.index.year
        df_daily['day_of_year'] = df_daily.index.dayofyear
        
        plt.figure(figsize=(12, 6))
        
        for year in df_daily['year'].unique():
            year_data = df_daily[df_daily['year'] == year]
            plt.plot(year_data['day_of_year'], year_data['unit_sales'], 
                    label=str(year), alpha=0.7)
        
        plt.title("Year-over-Year Sales Comparison")
        plt.xlabel("Day of Year")
        plt.ylabel("Unit Sales")
        plt.legend()
        plt.tight_layout()
        
        save_path = os.path.join(self.eda_path, f"{sales_col}_year_over_year.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"💾 Year-over-year plot saved to {save_path}")
        plt.show()

    def plot_autocorrelation(self, sales_col="unit_sales", lags=30):
        """Optimized: Uses daily aggregated data."""
        print(f"[DEBUG] Plotting autocorrelation...")  # noqa: F541
        
        plt.figure(figsize=(10, 5))
        # Use daily aggregated sales for ACF
        plot_acf(self.daily_sales.dropna(), lags=lags)
        plt.title(f"Autocorrelation of Daily Total Sales", fontsize=20, fontweight="bold")  # noqa: F541
        plt.tight_layout()
        
        save_path = os.path.join(self.eda_path, f"{sales_col}_autocorrelation.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"💾 Autocorrelation plot saved to {save_path}")
        plt.show()
    
    def check_and_plot_missing_calendar(self, df: pd.DataFrame = None, date_col="date") -> pd.DataFrame:
        """
        Check for missing calendar days in the dataset and plot coverage.
        Optimized: Uses unique dates only.
        """
        print("[DEBUG] Checking for missing calendar days...")
        
        # Use provided df or fall back to daily aggregated data
        if df is None:
            dates = self.daily_sales.index
        else:
            df[date_col] = pd.to_datetime(df[date_col])
            dates = df[date_col].unique()
        
        # Full date range
        min_date = dates.min()
        max_date = dates.max()
        all_dates = pd.date_range(start=min_date, end=max_date, freq="D")
        
        # Missing dates
        missing_dates = sorted(set(all_dates) - set(dates))
        coverage = (len(dates) / len(all_dates)) * 100
        
        # Build report
        report = pd.DataFrame({
            "total_days_in_range": [len(all_dates)],
            "days_with_data": [len(dates)],
            "missing_days": [len(missing_dates)],
            "data_coverage_percent": [coverage]
        })
        
        print(f"📅 Data coverage: {coverage:.2f}% ({len(missing_dates)} missing days)")
        
        # Plot coverage
        fig, ax = plt.subplots(figsize=(12, 4))
        date_series = pd.Series(1, index=all_dates)
        date_series[missing_dates] = 0
        ax.plot(date_series.index, date_series.values, 'o-', markersize=3)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(['Missing', 'Present'])
        ax.set_xlabel('Date')
        ax.set_title('Time Series Data Gaps (Missing Dates)')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = os.path.join(self.eda_path, "missing_calendar_days.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"💾 Missing calendar plot saved to {save_path}")
        plt.show()
        
        return report