import os
import pandas as pd  # type: ignore
import dask.dataframe as dd  # type: ignore
import pyarrow as pa  # type: ignore
import pyarrow.dataset as ds
import pyarrow.compute as pc
from datetime import datetime

from src.utils import get_path


class DataLoader:
    """
    Centralized loader for CSV files with:
    - Pandas loading
    - Dask loading
    - Bulk loading
    - Dataset metadata
    - PyArrow filtering
    - NEW: load_filtered_csv() for persistent filtered datasets
    """

    def __init__(self):
        self._cache: dict[str, pd.DataFrame] = {}
        print("🔧 DataLoader initialized with empty cache.")

    # ============================================================
    # Helper: Build deterministic filtered filename
    # ============================================================
    def _build_filtered_filename(self, table_name: str, filters: dict) -> str:
        """
        Build a deterministic filename based on filter parameters.
        Example:
        sales__MAXDATE-2014-04-01__STORE-24__ITEM-105577.csv
        """
        base = table_name.replace(".csv", "")
        parts = [base]

        if "MAX_DATE" in filters:
            clean_date = str(filters["MAX_DATE"]).split(" ")[0] 
            parts.append(f"MAXDATE-{clean_date}")

        if "STORE_IDS" in filters:
            parts.append("STORE-" + "-".join(map(str, filters["STORE_IDS"])))

        if "ITEM_IDS" in filters:
            parts.append("ITEM-" + "-".join(map(str, filters["ITEM_IDS"])))

        return "__".join(parts) + ".csv"

    # ============================================================
    # NEW METHOD: Load filtered CSV with persistent caching
    # ============================================================
    def load_filtered_csv(
        self,
        folder_name: str,
        table_name: str,
        filters: dict,
        filter_folder: str = "processed",
        week: int = None,
        force_recompute: bool = False,
        **kwargs
    ) -> pd.DataFrame:
        """
        Load a filtered version of a CSV.
        If filtered file exists → load it.
        If not → load raw CSV, apply filters, save filtered file, return it.

        Args:
            folder_name (str): Raw data folder (e.g., "raw")
            table_name (str): CSV filename
            filters (dict): Filtering rules:
                {
                    "MAX_DATE": "2014-04-01",
                    "STORE_IDS": [24],
                    "ITEM_IDS": [105577]
                }
            filter_folder (str): Folder where filtered files are stored
            week (int): Optional week parameter for get_path()
            force_recompute (bool): If True, ignore existing filtered file
            **kwargs: Passed to pandas.read_csv()

        Returns:
            pd.DataFrame: Filtered DataFrame
        """

        # Build filtered filename
        filtered_filename = self._build_filtered_filename(table_name, filters)
        filtered_path = os.path.join(get_path(filter_folder, week=week), filtered_filename)

        # ------------------------------------------------------------
        # 1. Load existing filtered file
        # ------------------------------------------------------------
        if os.path.exists(filtered_path) and not force_recompute:
            print(f"⚡ Loading existing filtered dataset: {filtered_filename}")
            df = pd.read_csv(filtered_path)
            print(f"✅ Loaded filtered dataset with shape: {df.shape}")
            if not df.empty:
                print("\n📅 Date Range:")
                print(f"   Start: {df['date'].min()}")
                print(f"   End:   {df['date'].max()}")
                print(f"   Days:  {len(df['date'].unique())}")
            return df

        print("🔍 No existing filtered file found. Will compute filtering...")

        # ------------------------------------------------------------
        # 2. Load raw CSV using existing load_csv()
        # ------------------------------------------------------------
        df = self.load_csv(
            folder_name=folder_name,
            table_name=table_name,
            week=1,
            use_cache=False,   # raw load only
            is_sample=False,
            **kwargs
        )
        
        if not df.empty:
            print("\n📅 Date Range:")
            print(f"   Start: {df['date'].min()}")
            print(f"   End:   {df['date'].max()}")
            print(f"   Days:  {len(df['date'].unique())}")

        # ------------------------------------------------------------
        # 3. Apply filters
        # ------------------------------------------------------------
        print("🔎 Applying filters...")

        # --- FIX: ensure date column is datetime ---
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")

        # --- Apply filters ---
        if "MAX_DATE" in filters and filters["MAX_DATE"] is not None:
            df = df[df["date"] <= pd.to_datetime(filters["MAX_DATE"])]


        if "STORE_IDS" in filters:
            df = df[df["store_nbr"].isin(filters["STORE_IDS"])]

        if "ITEM_IDS" in filters:
            df = df[df["item_nbr"].isin(filters["ITEM_IDS"])]

        print(f"✅ Filtered shape: {df.shape}")

        # ------------------------------------------------------------
        # 4. Save filtered dataset
        # ------------------------------------------------------------
        os.makedirs(os.path.dirname(filtered_path), exist_ok=True)
        df.to_csv(filtered_path, index=False)
        print(f"💾 Saved filtered dataset to: {filtered_path}")

        return df
    
    # ============================================================
    # NEW METHOD: Load filtered CSV by Max Date only
    # ============================================================
    def load_data_filtered_by_date(
        self,
        folder_name: str,
        table_name: str,
        max_date: str,
        filter_folder: str = "processed",
        week: int = None,
        force_recompute: bool = False,
        **kwargs
    ) -> pd.DataFrame:
        """
        Specialized version of load_filtered_csv focusing only on a date cutoff.
        Useful for creating snapshots or rolling window datasets.
        """
        
        # 1. Build a simple deterministic filename
        # Example: sales__MAXDATE-2014-04-01.csv
        clean_date = str(max_date).split(" ")[0]
        base_name = table_name.replace(".csv", "")
        filtered_filename = f"{base_name}__MAXDATE-{clean_date}.csv"
        
        filtered_path = os.path.join(get_path(filter_folder, week=week), filtered_filename)

        # 2. Check if already exists
        if os.path.exists(filtered_path) and not force_recompute:
            print(f"⚡ Loading existing date-filtered dataset: {filtered_filename}")
            df = pd.read_csv(filtered_path)
            print(f"Raw dataset range after after filtering with max_date: {max_date}")
            if not df.empty:
                print("\n📅 Date Range:")
                print(f"   Start: {df['date'].min()}")
                print(f"   End:   {df['date'].max()}")
                print(f"   Days:  {len(df['date'].unique())}")
            return df

        # 3. Load raw data
        print(f"🔍 Filtering {table_name} by Max Date: {clean_date}...")
        df = self.load_csv(
            folder_name=folder_name,
            table_name=table_name,
            week=week,
            use_cache=False,
            **kwargs
        )
        if not df.empty:
            print("\n📅 Date Range:")
            print(f"   Start: {df['date'].min()}")
            print(f"   End:   {df['date'].max()}")
            print(f"   Days:  {len(df['date'].unique())}")

        # 4. Apply Date Filter
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df = df[df["date"] <= pd.to_datetime(max_date)]
        else:
            print(f"⚠️ Warning: 'date' column not found in {table_name}. Returning unfiltered.")

        # 5. Save and Return
        os.makedirs(os.path.dirname(filtered_path), exist_ok=True)
        df.to_csv(filtered_path, index=False)
        print(f"💾 Saved date-filtered dataset (Shape: {df.shape})")
        
        return df

    # ============================================================
    # ORIGINAL load_csv (unchanged)
    # ============================================================
    def load_csv(
        self,
        folder_name: str,
        table_name: str,
        week: int = None,
        use_cache: bool = True,
        is_sample: bool = False,
        sample_size: int = 2_000_000,
        **kwargs
    ) -> pd.DataFrame:

        print(f"📂 Resolving path for folder: {folder_name}, week: {week}")
        path = get_path(folder_name, week=week)
        file_path = os.path.join(path, table_name)
        print(f"📄 Target file path: {file_path}")

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"❌ File '{table_name}' not found in {path}")

        cache_key = f"{folder_name}/{table_name}"
        if use_cache and cache_key in self._cache and not is_sample:
            print(f"⚡ Returning cached DataFrame for {cache_key}")
            return self._cache[cache_key]

        print(f"📖 Loading {table_name} with pandas...")

        if is_sample:
            df = pd.read_csv(file_path, nrows=sample_size, **kwargs)
        else:
            df = pd.read_csv(file_path, **kwargs)

        print(f"✅ Loaded {table_name} with shape {df.shape}")

        if use_cache and not is_sample:
            self._cache[cache_key] = df

        return df

    # ============================================================
    # Dask loader (unchanged)
    # ============================================================
    def load_csv_dask(self, folder_name: str, table_name: str, week: int = None,
                      is_sample: bool = False, sample_size: int = 2_000_000, **kwargs) -> dd.DataFrame:

        path = get_path(folder_name, week=week)
        file_path = os.path.join(path, table_name)
        print(f"📖 Lazily loading {table_name} with Dask...")

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"❌ File '{table_name}' not found in {path}")

        dask_df = dd.read_csv(file_path, **kwargs)

        if is_sample:
            dask_df = dask_df.sample(n=sample_size, random_state=42)

        print(f"✅ Dask DataFrame created for {table_name}")
        return dask_df

    # ============================================================
    # Bulk loader (unchanged)
    # ============================================================
    def load_all_csvs(self, folder_name: str, week: int = None,
                      use_cache: bool = True, is_sample: bool = False,
                      sample_size: int = 2_000_000, **kwargs) -> dict[str, pd.DataFrame]:

        path = get_path(folder_name, week=week)
        print(f"📁 Scanning folder: {path}")

        if not os.path.exists(path):
            raise FileNotFoundError(f"❌ Folder '{folder_name}' not found at {path}")

        csv_files = [f for f in os.listdir(path) if f.endswith(".csv")]
        dataframes = {}

        for f in csv_files:
            key = f.replace(".csv", "")
            dataframes[key] = self.load_csv(folder_name, f, use_cache=use_cache,
                                            is_sample=is_sample, sample_size=sample_size, **kwargs)

        return dataframes

    # ============================================================
    # Dataset info (unchanged)
    # ============================================================
    def get_dataset_info(self, folder_name: str = "raw", week: int = None,
                         sample_rows: int = 5) -> dict[str, dict]:

        path = get_path(folder_name, week=week)
        print(f"📁 Scanning folder: {path}")

        if not os.path.exists(path):
            raise FileNotFoundError(f"❌ Folder '{folder_name}' not found at {path}")

        csv_files = [f for f in os.listdir(path) if f.endswith(".csv")]
        info_dict = {}

        for f in csv_files:
            file_path = os.path.join(path, f)
            try:
                df = pd.read_csv(file_path, nrows=500)
                info_dict[f.replace(".csv", "")] = {
                    "filename": f,
                    "rows_sampled": len(df),
                    "columns": list(df.columns),
                    "dtypes": df.dtypes.astype(str).to_dict(),
                    "memory_usage_kb": int(df.memory_usage(deep=True).sum() / 1024),
                    "sample": df.head(sample_rows).to_dict(orient="records"),
                }
            except Exception as e:
                print(f"❌ Failed to read {f}: {e}")

        return info_dict

    # ============================================================
    # PyArrow loader with filtering (unchanged)
    # ============================================================
    def load_csv_with_pyarrow_filter(
        self,
        folder_name: str,
        table_name: str,
        filter_condition: str = None,
        filter_value: str = None,
        week: int = None,
        **kwargs
    ) -> pd.DataFrame:

        path = get_path(folder_name, week=week)
        file_path = os.path.join(path, table_name)

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"❌ File '{table_name}' not found in {path}")

        if filter_condition is None or filter_value is None:
            raise ValueError("Filter condition and value must be provided.")

        dataset = ds.dataset(file_path, format="csv", **kwargs)
        date_obj = datetime.strptime(filter_value, "%Y-%m-%d").date()
        filter_scalar = pa.scalar(date_obj, type=pa.date32())

        table = dataset.to_table(filter=pc.field(filter_condition) <= filter_scalar)
        df = table.to_pandas()

        return df

    