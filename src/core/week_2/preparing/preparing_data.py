# src/core/week_2/preparing_data.py
"""Data preparation module for time series forecasting models.
Includes filtering, aggregation, calendar completion, and conversions
to formats suitable for ARIMA, SARIMA, ETS, Prophet, and Darts.
"""
import os
from typing import List, Optional, Tuple

import pandas as pd  # type: ignore
import numpy as np   # type: ignore  # noqa: F401
import matplotlib.pyplot as plt  # type: ignore
from darts import TimeSeries  # type: ignore

from src.utils import get_path
from src.core.week_1.loader.loader import DataLoader


class PreparingData:
    """
    Unified data preparation for ARIMA, SARIMA, ETS, Prophet, and Darts workflows.
    Produces a clean dense daily series plus optional exogenous regressors.
    """

    def __init__(
        self,
        store_ids: Optional[List[int]] = None,
        item_ids: Optional[List[int]] = None,
        max_date: Optional[str] = None,
        folder_name: str = "features",                     # NEW DEFAULT
        table_name: str = "final_train_dataset.csv",       # NEW DEFAULT
        filter_folder: str = "filtered", 
        date_col: str = "date",
        store_col: str = "store_nbr",
        item_col: str = "item_nbr",
        sales_col: str = "unit_sales",
        promo_col: Optional[str] = "onpromotion",
        verbose: bool = True
    ):
        self.store_ids = store_ids or [24]
        self.item_ids = item_ids or [105577]
        self.max_date = pd.to_datetime(max_date) if max_date else None

        self.folder_name = folder_name
        self.table_name = table_name
        self.filter_folder = filter_folder

        self.date_col = date_col
        self.store_col = store_col
        self.item_col = item_col
        self.sales_col = sales_col
        self.promo_col = promo_col
        self.verbose = verbose
        
        self.loader = DataLoader()

        # artifacts
        self.df_filtered: Optional[pd.DataFrame] = None
        self.df_daily: Optional[pd.DataFrame] = None
        self.df_calendar: Optional[pd.DataFrame] = None
        self.series: Optional[TimeSeries] = None
        self.train: Optional[TimeSeries] = None
        self.test: Optional[TimeSeries] = None

    def _log(self, msg: str):
        if self.verbose:
            print(f"[PreparingData] {msg}")

    # --------------------------------------------------------- 
    #   Load + filter using DataLoader 
    # ---------------------------------------------------------
    
    def load_and_filter(self, week: int = 1) -> pd.DataFrame:

        filters = {
            "MAX_DATE": self.max_date,
            "STORE_IDS": self.store_ids,
            "ITEM_IDS": self.item_ids
        }

        self._log("Loading filtered dataset via DataLoader...")

        df = self.loader.load_filtered_csv(
            folder_name=self.folder_name,
            table_name=self.table_name,
            filters=filters,
            filter_folder=self.filter_folder,
            week=week
        )

        df[self.date_col] = pd.to_datetime(df[self.date_col], errors="coerce")
        self.df_filtered = df

        self._log(f"✓ Loaded filtered dataset. Shape={df.shape}")
        return df


    def aggregate_daily(self) -> pd.DataFrame:
        if self.df_filtered is None or self.df_filtered.empty:
            raise ValueError("Run filter_single_series first and ensure non-empty result.")
        df = self.df_filtered.copy()
        agg_dict = {self.sales_col: "sum"}
        if self.promo_col and self.promo_col in df.columns:
            agg_dict[self.promo_col] = "max"
        df = df.groupby(self.date_col).agg(agg_dict).reset_index()
        df.set_index(self.date_col, inplace=True)
        df.sort_index(inplace=True)
        self.df_daily = df
        self._log(f"✓ Daily aggregation complete. Shape={df.shape}")
        return self.df_daily

    def fill_calendar(self) -> pd.DataFrame:
        if self.df_daily is None or self.df_daily.empty:
            raise ValueError("Run aggregate_daily first and ensure non-empty result.")
        start, end = self.df_daily.index.min(), self.df_daily.index.max()
        full_range = pd.date_range(start, end, freq="D")
        self.df_calendar = self.df_daily.reindex(full_range, fill_value=0)
        self.df_calendar.index.name = self.date_col
        self._log(f"✓ Calendar completed. Final shape={self.df_calendar.shape}")
        return self.df_calendar

    # -----------------------------
    # Model adapters
    # -----------------------------
    def to_timeseries(self) -> TimeSeries:
        """Convert the completed calendar DataFrame into a Darts TimeSeries."""
        if self.df_calendar is None:
            raise ValueError("Run fill_calendar first.")

        self.series = TimeSeries.from_dataframe(
            self.df_calendar.reset_index(),
            time_col=self.date_col,
            value_cols=self.sales_col,
            fill_missing_dates=True,
            freq="D"
        )

        self._log(f"✓ Darts TimeSeries created. Length={len(self.series)}")
        return self.series


    def to_arima_input(self) -> pd.Series:
        """Pandas Series with DateTimeIndex for ARIMA/SARIMA/ETS."""
        if self.df_calendar is None:
            raise ValueError("Run fill_calendar first.")
        return self.df_calendar[self.sales_col]

    def to_exog(self, cols: Optional[List[str]] = None) -> Optional[pd.DataFrame]:
        """Exogenous regressors for SARIMAX/Prophet if available."""
        if self.df_calendar is None:
            raise ValueError("Run fill_calendar first.")
        if cols is None:
            cols = [c for c in ["onpromotion", "days_until_holiday", "days_since_holiday"] if c in self.df_calendar.columns]
        return self.df_calendar[cols] if cols else None

    def to_prophet_input(self) -> pd.DataFrame:
        """DataFrame with ds/y for Prophet."""
        if self.df_calendar is None:
            raise ValueError("Run fill_calendar first.")
        return self.df_calendar.reset_index()[[self.date_col, self.sales_col]].rename(
            columns={self.date_col: "ds", self.sales_col: "y"}
        )

    def to_ets_input(self) -> pd.Series:
        """Alias to ARIMA input for ETS."""
        return self.to_arima_input()

    # -----------------------------
    # Splits
    # -----------------------------
    def split_train_test(self, split_ratio: float = 0.8) -> Tuple[TimeSeries, TimeSeries]:
        """
        Darts TimeSeries split (chronological).
        """
        if self.series is None:
            raise ValueError("Call to_timeseries first.")
        if len(self.series) == 0:
            raise ValueError("TimeSeries is empty.")
        self._log(f"Splitting TimeSeries: train={int(split_ratio*100)}% test={int((1-split_ratio)*100)}%")
        try:
            train, test = self.series.split_after(split_ratio)
            self.train, self.test = train, test
            self._log(f"✓ Train len={len(train)}, Test len={len(test)}")
            self._log(f"✓ Train range: {train.start_time().date()} → {train.end_time().date()}")
            self._log(f"✓ Test range: {test.start_time().date()} → {test.end_time().date()}")
            time_series_path = get_path("time_series")
            os.makedirs(time_series_path, exist_ok=True)
            save_train_path = os.path.join(time_series_path, "train_series.csv")
            save_test_path = os.path.join(time_series_path, "test_series.csv")
            train.to_csv(save_train_path, index=False)
            test.to_csv(save_test_path, index=False)
            self._log(f"💾 Train TimeSeries saved to {save_train_path}")
            self._log(f"💾 Test TimeSeries saved to {save_test_path}")
            return train, test
        except Exception as e:
            self._log(f"❌ Error splitting TimeSeries: {e}")
            raise

    def split_series_train_test(self, split_ratio: float = 0.8) -> Tuple[pd.Series, pd.Series]:
        """
        Pandas Series split for ARIMA/SARIMA/ETS.
        """
        y = self.to_arima_input()
        n = len(y)
        if n == 0:
            raise ValueError("Series is empty.")
        cutoff = int(n * split_ratio)
        y_train, y_test = y.iloc[:cutoff], y.iloc[cutoff:]
        self._log(f"✓ Series split: train={len(y_train)} test={len(y_test)}")
        return y_train, y_test

    def split_prophet_train_test(self, split_ratio: float = 0.8) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Prophet DataFrame split using ds chronological order.
        """
        dfp = self.to_prophet_input().sort_values("ds")
        n = len(dfp)
        if n == 0:
            raise ValueError("Prophet DataFrame is empty.")
        cutoff = int(n * split_ratio)
        train, test = dfp.iloc[:cutoff].copy(), dfp.iloc[cutoff:].copy()
        self._log(f"✓ Prophet split: train={len(train)} test={len(test)}")
        return train, test


    def visualize(self, title: str = "Daily unit_sales", figsize: Tuple[int, int] = (21, 7)):
        """Quick plot of the current TimeSeries for visual inspection."""
        if self.series is None:
            raise ValueError("Call to_timeseries first.")
        
        self._log("Plotting series...")
        plt.figure(figsize=figsize)
        self.series.plot()
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

   

    # -----------------------------
    # Helper methods for debugging
    # -----------------------------
    
    def get_data_summary(self) -> dict:
        """Get summary of current data state."""
        summary = {
            "store_ids": self.store_ids,
            "item_ids": self.item_ids,
            "max_date": self.max_date.date() if self.max_date else None,
            "df_filtered_shape": self.df_filtered.shape if self.df_filtered is not None else None,
            "df_daily_shape": self.df_daily.shape if self.df_daily is not None else None,
            "df_calendar_shape": self.df_calendar.shape if self.df_calendar is not None else None,
            "series_length": len(self.series) if self.series is not None else None,
            "train_length": len(self.train) if self.train is not None else None,
            "test_length": len(self.test) if self.test is not None else None,
        }
        
        if self.df_filtered is not None and not self.df_filtered.empty:
            summary["filtered_date_range"] = f"{self.df_filtered[self.date_col].min().date()} → {self.df_filtered[self.date_col].max().date()}"
        
        if self.df_daily is not None and not self.df_daily.empty:
            summary["daily_date_range"] = f"{self.df_daily.index.min().date()} → {self.df_daily.index.max().date()}"
        
        return summary
    
    def print_data_summary(self):
        """Print summary of current data state."""
        summary = self.get_data_summary()
        print("\n" + "="*60)
        print("DATA PREPARATION SUMMARY")
        print("="*60)
        for key, value in summary.items():
            print(f"{key:30}: {value}")
        print("="*60)
    
    def validate_data_pipeline(self) -> bool:
        """
        Validate the entire data pipeline.
        Returns True if all steps are successful, False otherwise.
        """
        try:
            steps = [
                ("Filtered data exists", self.df_filtered is not None and not self.df_filtered.empty),
                ("Daily aggregation exists", self.df_daily is not None and not self.df_daily.empty),
                ("Calendar completion exists", self.df_calendar is not None and not self.df_calendar.empty),
                ("TimeSeries exists", self.series is not None and len(self.series) > 0),
            ]
            
            print("\n" + "="*60)
            print("DATA PIPELINE VALIDATION")
            print("="*60)
            
            all_valid = True
            for step_name, is_valid in steps:
                status = "✓" if is_valid else "❌"
                print(f"{status} {step_name}")
                if not is_valid:
                    all_valid = False
            
            if all_valid:
                print("\n✓ All validation checks passed!")
            else:
                print("\n⚠ Some validation checks failed.")
            
            return all_valid
            
        except Exception as e:
            print(f"❌ Validation error: {e}")
            return False