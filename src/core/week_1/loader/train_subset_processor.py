
# ===========================================================================
# COMPLETE PIPELINE: Train Subset Processor + Runner Script
# ===========================================================================

# ---------------------------------------------------------------------------
# FILE 1: src/week_1/processor/train_subset_processor.py
# ---------------------------------------------------------------------------

import json
import os
import pandas as pd  # type: ignore
from IPython.display import display # type: ignore

from src.utils import get_path
from .visualizer import Visualizer
from .loader import DataLoader


class TrainSubsetProcessor:
    """
    Processor for preparing train subsets from Favorita dataset.
    Provides filtering by region, top families, chunked reading, sampling, and saving.
    """

    def __init__(self, loader=None):
        """
        Args:
            loader: DataLoader-compatible object.
        """
        self.loader = loader if loader is not None else DataLoader()
        self.visualizer = Visualizer()
        self.region = "Guayas"  # default region
        self.df_items = self._get_item()  # Cache items dataframe
        self.df_subset = None   
        
    def _get_item(self, item_file: str = "items.csv") -> pd.DataFrame:
        """Load items data."""
        df_items = self.loader.load_csv("raw", item_file)
        return df_items    

    def get_region_stores(self, region: str, store_file: str = "stores.csv") -> list[int]:
        """Get store IDs for a specific region."""
        df_stores = self.loader.load_csv("raw", store_file)
        store_ids = df_stores[df_stores["state"] == region]["store_nbr"].unique().tolist()
        print(f"✅ Selected {len(store_ids)} stores from region '{region}'.")
        return store_ids

    def get_top_families(self, item_file: str = "items.csv", top_n: int = 3) -> list[str]:
        """Get top N product families by item count."""
        self.df_items = self.loader.load_csv("raw", item_file)
        top_families = self.df_items["family"].value_counts().nlargest(top_n).index.tolist()
        print(f"🎯 Top {top_n} product families: {top_families}")
        
        self.visualizer.plot_distribution(
            df=self.df_items, 
            region=self.region, 
            top_families=top_families,
            items_df=self.df_items
        )
        return top_families

    def filter_train_chunks(self, store_ids, train_file="train.csv", chunk_size=10**6) -> pd.DataFrame:
        """Read train.csv in chunks and filter by store IDs."""
        train_path = os.path.join(get_path("raw"), train_file)
        if not os.path.exists(train_path):
            raise FileNotFoundError(f"❌ {train_file} not found in raw folder.")

        filtered_chunks = []
        total_rows_read = 0
        print(f"📖 Reading {train_file} in chunks of {chunk_size:,} rows...")
        
        for i, chunk in enumerate(pd.read_csv(train_path, chunksize=chunk_size), 1):
            total_rows_read += len(chunk)
            chunk_filtered = chunk[chunk["store_nbr"].isin(store_ids)]
            
            if not chunk_filtered.empty:
                filtered_chunks.append(chunk_filtered)
                print(f"   Chunk {i}: {len(chunk_filtered):,} rows matched (total read: {total_rows_read:,})")
            del chunk

        if not filtered_chunks:
            raise ValueError(f"❌ No rows found for stores {store_ids}.")
        
        df_train = pd.concat(filtered_chunks, ignore_index=True)
        del filtered_chunks
        
        print(f"✅ Combined {len(df_train):,} filtered rows from all chunks.")
        print(f"   Final shape: {df_train.shape}")
        return df_train

    def filter_top_families(self, df_train: pd.DataFrame, item_file: str = "items.csv", top_n: int = 3) -> pd.DataFrame:
        """Filter train data to only include top N product families."""
        if self.df_items is None:
            self.df_items = self.loader.load_csv("raw", item_file)
        
        items_per_family = self.df_items["family"].value_counts().reset_index()
        items_per_family.columns = ["Family", "Item Count"]
        top_families = items_per_family.head(top_n)
        print(f"🎯 Top {top_n} families: {top_families['Family'].tolist()}")
        
        item_ids = self.df_items[
            self.df_items["family"].isin(top_families["Family"])
        ]["item_nbr"].unique()
        
        df_train_filtered = df_train[df_train["item_nbr"].isin(item_ids)].copy()
        print(f"✅ Filtered to top {top_n} families.")
        print(f"   Before: {len(df_train):,} rows")
        print(f"   After: {len(df_train_filtered):,} rows")
        
        self.visualizer.plot_distribution(
            df=df_train_filtered,
            region=self.region,
            top_families=top_families["Family"].tolist(),
            items_df=self.df_items
        )
        return df_train_filtered

    def sample_subset(self, df, sample_size: int) -> pd.DataFrame:
        """Sample a subset of rows from the DataFrame."""
        if sample_size < len(df):
            df_sample = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
            print(f"🎯 Sampled {sample_size:,} rows from {len(df):,} available rows.")
            print(f"   Sample rate: {sample_size/len(df)*100:.1f}%")
        else:
            print(f"⚠️ Requested sample_size ({sample_size:,}) >= available rows ({len(df):,}).")
            print(f"   Using full dataset without sampling.")  # noqa: F541
            df_sample = df.copy()
        return df_sample

    def prepare_train_subset(
        self,
        region: str = "Guayas",
        sample_size: int = 2_000_000,
        chunk_size: int = 10**6,
        top_n: int = 3,
        output_name: str = "train_subset.csv",
        store_file: str = "stores.csv",
        item_file: str = "items.csv",
        train_file: str = "train.csv",
        plot_workflow: bool = True
    ) -> pd.DataFrame:
        """
        Prepare a subset of the large train.csv file.
        
        Pipeline Steps:
        1. Filter by region stores
        2. Read train.csv in chunks
        3. Filter to top N families
        4. Sample to target size
        5. Save to processed folder
        """
        print("\n" + "="*70)
        print("🚀 TRAIN SUBSET PREPARATION PIPELINE")
        print("="*70)
        print(f"Configuration:")  # noqa: F541
        print(f"  • Region: {region}")
        print(f"  • Top Families: {top_n}")
        print(f"  • Chunk Size: {chunk_size:,} rows")
        print(f"  • Target Sample: {sample_size:,} rows")
        print(f"  • Output: {output_name}")
        print("="*70 + "\n")
        
        self.region = region
        
        # Step 1: Region filter
        print("📍 STEP 1/5: Filtering by region stores...")
        print("-" * 70)
        store_ids = self.get_region_stores(region, store_file)
        print()
        
        # Step 2: Chunked filtering
        print("📖 STEP 2/5: Reading and filtering train.csv by stores...")
        print("-" * 70)
        df_train = self.filter_train_chunks(store_ids, train_file, chunk_size)
        print()
        
        # Step 3: Filter families
        print(f"🎯 STEP 3/5: Filtering to top {top_n} product families...")
        print("-" * 70)
        df_train = self.filter_top_families(df_train, item_file, top_n)
        print()
        
        # Step 4: Sampling
        # print("🎲 STEP 4/5: Sampling final subset...")
        # print("-" * 70)
        # df_train = self.sample_subset(df_train, sample_size)
        # print()
        
        # Step 5: Save subset
        print("💾 STEP 5/5: Saving processed data...")
        print("-" * 70)
        output_path = os.path.join(get_path("loader_processed", week=1), output_name)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df_train.to_csv(output_path, index=False)
        
        file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"✅ Saved processed train subset to:")  # noqa: F541
        print(f"   {output_path}")
        print(f"   Size: {file_size_mb:.2f} MB")
        print()
        
        # Step 6: Generate workflow
        if plot_workflow:
            print("📊 STEP 6/6: Generating workflow diagram...")
            print("-" * 70)
            self.visualizer.plot_train_subset_workflow(
                region=region,
                sample_size=sample_size,
                chunk_size=chunk_size,
                top_n=top_n
            )
            print()
        
        # Final summary
        print("="*70)
        print("✅ PIPELINE COMPLETE!")
        print("="*70)
        print(f"Summary:")  # noqa: F541
        print(f"  • Region: {region} ({len(store_ids)} stores)")
        print(f"  • Top {top_n} families filtered")
        print(f"  • Final rows: {len(df_train):,}")
        print(f"  • Columns: {len(df_train.columns)}")
        print(f"  • Memory: {df_train.memory_usage(deep=True).sum() / (1024**2):.2f} MB")
        print("="*70 + "\n")
        self.df_subset = df_train
        return df_train

    def transform_to_daily(
        self,
        df_train: pd.DataFrame,
        group_by: str = "family",
        agg_func: str = "sum",
        week: int = 1,
        plot_daily: bool = True,
        top_n: int = 3
    ) -> pd.DataFrame:
        """
        Transform filtered train subset into daily time series.
        
        Args:
            df_train: Subset of train data
            group_by: Dimension to group by ('family', 'store_nbr', 'item_nbr')
            agg_func: Aggregation function ('sum' or 'mean')
            week: Week number for path
            plot_daily: Whether to generate daily plot
            top_n: Number of top groups to keep (applies only to store_nbr and item_nbr)
            
        Returns:
            pd.DataFrame: Daily aggregated dataset
        """
        output_name = f"train_daily_{group_by}.csv"
        print("\n" + "="*70)
        print("📅 DAILY TIME SERIES TRANSFORMATION")
        print("="*70)
        print(f"Configuration:")  # noqa: F541
        print(f"  • Group by: {group_by}")
        print(f"  • Aggregation: {agg_func}")
        print(f"  • Output: {output_name}")
        print("-" * 70)

        # Merge with items to get family if needed
        if group_by == "family" and "family" not in df_train.columns:
            print("  ℹ️  Merging with items to get family column...")
            df_train = df_train.merge(
                self.df_items[['item_nbr', 'family']], 
                on='item_nbr', 
                how='left'
            )

        # Ensure date is datetime
        df_train["date"] = pd.to_datetime(df_train["date"])

        # Restrict to top N only for store_nbr and item_nbr
        if group_by in ["store_nbr", "item_nbr"]:
            top_groups = (
                df_train[group_by]
                .value_counts()
                .nlargest(top_n)
                .index
                .tolist()
            )
            df_train = df_train[df_train[group_by].isin(top_groups)]
            print(f"  🎯 Restricted to top {top_n} {group_by} values: {top_groups}")

        # Group and aggregate
        df_daily = (
            df_train.groupby(["date", group_by])["unit_sales"]
            .agg(agg_func)
            .reset_index()
            .sort_values("date")
        )

        print(f"✅ Transformed to daily time series.")  # noqa: F541
        print(f"   Shape: {df_daily.shape}")
        print(f"   Date range: {df_daily['date'].min()} to {df_daily['date'].max()}")
        print(f"   Unique {group_by}: {df_daily[group_by].nunique()}")

        # Save
        output_path = os.path.join(get_path("loader_processed", week=week), output_name)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df_daily.to_csv(output_path, index=False)
        print(f"💾 Daily dataset saved to {output_path}")

        # Plot if requested
        if plot_daily:
            print(f" Generating daily time series plot...")  # noqa: F541
            self.visualizer.plot_daily(df_daily, group_by=group_by)

        print("="*70 + "\n")
        return df_daily
    
    def fill_missing_daily_sales(
        self,
        df_train: pd.DataFrame,
        week: int = 1,
        save: bool = True,
        plot: bool = False
    ) -> pd.DataFrame:
        """
        Ensure every (store_nbr, item_nbr) has a complete daily time series.
        Missing dates are filled with zero sales.

        Args:
            df_train (pd.DataFrame): Raw training subset with at least ['date','store_nbr','item_nbr','unit_sales'].
            week (int): Week number for saving path.
            save (bool): Whether to save the filled dataset to CSV.
            plot (bool): Whether to plot daily coverage using FeatureViz.

        Returns:
            pd.DataFrame: Dense daily panel with zero-filled sales.
        """
        print("\n" + "="*70)
        print("📅 FILL MISSING DAILY SALES")
        print("="*70)

        # --- BEFORE SNAPSHOT ---
        print("🔍 Input snapshot Data of Guayas Region:")
        print(f"   • Rows: {len(df_train)}")
        print(f"   • Columns: {list(df_train.columns)}")
        print(f"   • Date range: {df_train['date'].min()} → {df_train['date'].max()}")
        print(f"   • Unique stores: {df_train['store_nbr'].nunique()}")
        print(f"   • Unique items: {df_train['item_nbr'].nunique()}")
        print("-" * 70)

        # Ensure datetime
        df_train["date"] = pd.to_datetime(df_train["date"])

        # Build full date range
        min_date, max_date = df_train["date"].min(), df_train["date"].max()
        full_date_range = pd.DataFrame({"date": pd.date_range(min_date, max_date, freq="D")})
        print(f" Created full date range: {min_date} to {max_date} ({len(full_date_range)} days)")

        # Build all store–item combinations
        store_item_combinations = df_train[["store_nbr", "item_nbr"]].drop_duplicates()
        all_combinations = store_item_combinations.merge(full_date_range, how="cross")
        print(f" Created all store–item–date combinations: {len(all_combinations):,} rows")

        # Merge back and fill missing sales with 0
        df_filled = all_combinations.merge(
            df_train, on=["store_nbr", "item_nbr", "date"], how="left"
        )
        df_filled["unit_sales"] = df_filled["unit_sales"].fillna(0)

        # --- AFTER SNAPSHOT ---
        print("✅ Completed filling missing dates.")
        print(f"   • Final shape: {df_filled.shape}")
        print(f"   • Date range: {df_filled['date'].min()} → {df_filled['date'].max()}")
        print(f"   • Unique stores: {df_filled['store_nbr'].nunique()}")
        print(f"   • Unique items: {df_filled['item_nbr'].nunique()}")
        print(f"   • Columns: {list(df_filled.columns)}")
        print(f"   • Memory usage: {df_filled.memory_usage(deep=True).sum() / (1024**2):.2f} MB")

        df_filled['id'] = df_filled['id'].fillna(0).astype(int)
        # Save
        if save:
            output_name = f"train_daily_filled.csv"  # noqa: F541
            output_path = os.path.join(get_path("loader_processed", week=week), output_name)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            df_filled.to_csv(output_path, index=False)
            print(f"💾 Filled dataset saved to {output_path}")

        # Optional visualization
        if plot:
            self.viz.plot_daily(df_filled, group_by="store_nbr")

        # --- PIPELINE COMPLETE BANNER ---
        print("="*70)
        print("✅ PIPELINE COMPLETE!")
        print("="*70)
        print("Summary:")
        print(f"  • Final rows: {len(df_filled):,}")
        print(f"  • Columns: {df_filled.shape[1]}")
        print(f"  • Memory: {df_filled.memory_usage(deep=True).sum() / (1024**2):.2f} MB")
        print("="*70 + "\n")

        return df_filled


    def save_pipeline_stats(self, stats: dict, week: int = 1, 
                           filename: str = "pipeline_stats.json") -> str:
        """Save pipeline statistics to JSON."""
        output_dir = get_path("eda_stats", week=week)
        output_path = os.path.join(output_dir, filename)
        os.makedirs(output_dir, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=4, default=str)

        print(f"💾 Pipeline statistics saved to {output_path}")
        return output_path

    def get_pipeline_stats(self, df: pd.DataFrame) -> dict:
        """Get statistics about the processed dataset."""
        stats = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'memory_mb': df.memory_usage(deep=True).sum() / (1024**2),
            'date_range': (str(df['date'].min()), str(df['date'].max())) if 'date' in df.columns else None,
            'unique_stores': int(df['store_nbr'].nunique()) if 'store_nbr' in df.columns else None,
            'unique_items': int(df['item_nbr'].nunique()) if 'item_nbr' in df.columns else None,
            'total_sales': float(df['unit_sales'].sum()) if 'unit_sales' in df.columns else None,
            'missing_values': int(df.isnull().sum().sum())
        }
        self.save_pipeline_stats(stats, week=1)
        return stats

    def print_stats(self, df: pd.DataFrame):
        """Print formatted statistics about the dataset."""
        stats = self.get_pipeline_stats(df)
        
        print("\n" + "="*70)
        print("📊 DATASET STATISTICS")
        print("="*70)
        print(f"Dimensions:")  # noqa: F541
        print(f"  • Rows: {stats['total_rows']:,}")
        print(f"  • Columns: {stats['total_columns']}")
        print(f"  • Memory: {stats['memory_mb']:.2f} MB")
        print()
        
        if stats['date_range']:
            print(f"Date Range:")  # noqa: F541
            print(f"  • Start: {stats['date_range'][0]}")
            print(f"  • End: {stats['date_range'][1]}")
            print()
        
        if stats['unique_stores']:
            print(f"Coverage:")  # noqa: F541
            print(f"  • Unique Stores: {stats['unique_stores']:,}")
        if stats['unique_items']:
            print(f"  • Unique Items: {stats['unique_items']:,}")
        if stats['total_sales']:
            print(f"  • Total Sales: {stats['total_sales']:,.0f} units")
        print()
        
        print(f"Data Quality:")  # noqa: F541
        print(f"  • Missing Values: {stats['missing_values']:,}")
        print(f"  • Completeness: {(1 - stats['missing_values']/(stats['total_rows']*stats['total_columns']))*100:.2f}%")
        print("="*70 + "\n")
    
    def run_pipeline(self):
        """
        Main execution function for Week 1 pipeline.

        Steps:
        1. Prepare train subset (filter region + families)
        2. Transform to daily time series (by family, store, item)
        3. Generate statistics and visualizations
        4. Return all processed datasets and stats
        """

        print("\n" + "="*70)
        print("🌟 FAVORITA STORE SALES PIPELINE - WEEK 1")
        print("="*70)
        print("Author: Data Science Team")
        print("Purpose: Prepare train subset and daily time series")
        print("="*70 + "\n")

        # ========================================================================
        # STEP 1: PREPARE TRAIN SUBSET
        # ========================================================================
        print("📦 STEP 1: Preparing train subset...")
        print("="*70)

        # df_subset = self.prepare_train_subset(
        #     region="Guayas",              # Target region
        #     output_name="train_subset_guayas.csv",
        #     plot_workflow=True            # Generate workflow diagram
        # )
        if self.df_subset is not None:
            
            df_subset = self.df_subset
            print(f"✅ Train subset ready with {len(df_subset):,} rows.")
            # Print statistics
            self.print_stats(df_subset)

            # ========================================================================
            # STEP 2: TRANSFORM TO DAILY (BY FAMILY)
            # ========================================================================
            print("📅 STEP 2: Transforming to daily time series (by family)...")
            print("="*70)

            df_daily_family = self.transform_to_daily(
                df_train=df_subset,
                group_by="family",
                agg_func="sum",
                week=1,
                plot_daily=True
            )

            print(f"✅ Daily dataset (by family) created: {len(df_daily_family)} rows")
            display(df_daily_family.sample(10))
            print()

            # ========================================================================
            # STEP 3: TRANSFORM TO DAILY (BY STORE)
            # ========================================================================
            print("📅 STEP 3: Transforming to daily time series (by store)...")
            print("="*70)

            df_daily_store = self.transform_to_daily(
                df_train=df_subset,
                group_by="store_nbr",
                agg_func="sum",
                week=1,
                plot_daily=True
            )

            print(f"✅ Daily dataset (by store) created: {len(df_daily_store)} rows")
            display(df_daily_store.sample(10))
            print()

            # ========================================================================
            # STEP 4: TRANSFORM TO DAILY (BY ITEM)
            # ========================================================================
            print("📅 STEP 4: Transforming to daily time series (by item)...")
            print("="*70)

            df_daily_item = self.transform_to_daily(
                df_train=df_subset,
                group_by="item_nbr",
                agg_func="sum",
                week=1,
                plot_daily=True
            )

            print(f"✅ Daily dataset (by item) created: {len(df_daily_item)} rows")
            display(df_daily_item.sample(10))
            print()

            # ========================================================================
            # STEP 5: PREPROCESSING VISUALIZATION (OPTIONAL)
            # ========================================================================
            print("📊 STEP 5: Generating preprocessing visualization...")
            print("="*70)

            # Use first family for demonstration
            first_family = df_daily_family[df_daily_family.groupby('family')['unit_sales'].transform('sum') > 0]['family'].iloc[0]
            df_sample = df_daily_family[df_daily_family['family'] == first_family].copy()

            if len(df_sample) > 0:
                print(f"   Using family '{first_family}' for preprocessing demo")
                self.visualizer.plot_preprocessing_steps(df_sample)
            else:
                print("   ⚠️ Skipping preprocessing plot (insufficient data)")

            # ========================================================================
            # FINAL SUMMARY
            # ========================================================================
            print("\n" + "="*70)
            print("🎉 PIPELINE EXECUTION COMPLETE!")
            print("="*70)
            print("\n📁 Output Files Created:")
            print("   1. train_subset_guayas.csv - Filtered train subset")
            print("   2. train_daily_by_family.csv - Daily aggregated by family")
            print("   3. train_daily_by_store.csv - Daily aggregated by store")
            print("   4. train_daily_by_item.csv - Daily aggregated by item")
            print("   5. pipeline_stats.json - Execution statistics")
            print("\n📊 Visualizations Created:")
            print("   1. distribution_Guayas.png - Store & family distribution")
            print("   2. train_subset_workflow_detailed.png - Pipeline flowchart")
            print("   3. daily_unit_sales_by_family.png - Time series plot")
            print("   4. daily_unit_sales_by_store_nbr.png - Time series plot")
            print("   5. daily_unit_sales_by_item_nbr.png - Time series plot")
            print("   6. preprocessing_steps.png - Preprocessing demo")
            print("\n✨ Next Steps:")
            print("   → Review the generated visualizations")
            print("   → Check data quality in the CSV files")
            print("   → Proceed to Week 2 (Time Series Analysis)")
            print("="*70 + "\n")

            # Return all datasets and stats for programmatic use
            #stats = self.get_pipeline_stats(df_subset)
            return {
                "subset": df_subset,
                "daily_family": df_daily_family,
                "daily_store": df_daily_store,
                "daily_item": df_daily_item,
                #"stats": stats
            }
        else:
            raise ValueError("❌ No train subset available. Please run prepare_train_subset() first.")








