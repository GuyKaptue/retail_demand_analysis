# src/core/week_1/main_week1.py

from .loader import DataLoader
from .loader import TrainSubsetProcessor
from .processor import EDAReportGenerator
from .processor import DataPreparationPipeline

def main():
    print("="*80)
    print("🌟 FAVORITA WEEK 1 MAIN PIPELINE")
    print("="*80)

    # -----------------------------------------------------------------------
    # STEP 0: Load raw data (already downloaded via Kaggle/GoogleDrive loaders)
    # -----------------------------------------------------------------------
    loader = DataLoader()
    print("📖 Loading raw CSVs...")
    raw_data = loader.load_all_csvs("raw", week=1)
    print(f"✅ Loaded {len(raw_data)} raw files")

    # -----------------------------------------------------------------------
    # STEP 1: Train subset preparation
    # -----------------------------------------------------------------------
    processor = TrainSubsetProcessor(loader=loader)
    df_subset = processor.prepare_train_subset(
        region="Guayas",
        sample_size=2_000_000,
        chunk_size=10**6,
        top_n=3,
        output_name="train_subset_guayas.csv",
        plot_workflow=True
    )

    # -----------------------------------------------------------------------
    # STEP 2: Daily transformations
    # -----------------------------------------------------------------------
    df_daily_family = processor.transform_to_daily(df_subset, group_by="family", agg_func="sum", week=1)  # noqa: F841
    df_daily_store  = processor.transform_to_daily(df_subset, group_by="store_nbr", agg_func="sum", week=1)  # noqa: F841
    df_daily_item   = processor.transform_to_daily(df_subset, group_by="item_nbr", agg_func="sum", week=1)  # noqa: F841

    # -----------------------------------------------------------------------
    # STEP 3: EDA report
    # -----------------------------------------------------------------------
    eda = EDAReportGenerator(df_subset, week=1)
    eda_report = eda.run_full_eda()  # noqa: F841

    # -----------------------------------------------------------------------
    # STEP 4: Feature engineering + impact analysis
    # -----------------------------------------------------------------------
    holidays_df = raw_data.get("holiday_events")
    oil_df = raw_data.get("oil")

    prep = DataPreparationPipeline(df_subset, holidays_df=holidays_df, oil_df=oil_df, week=1)
    prep.run_pipeline()

    # -----------------------------------------------------------------------
    # FINAL SUMMARY
    # -----------------------------------------------------------------------
    print("="*80)
    print("🎉 WEEK 1 PIPELINE COMPLETE")
    print("="*80)
    print("Outputs generated in:")
    print("  • processed/loader_processed/week_1/")
    print("  • eda/week_1/")
    print("  • features/week_1/")
    print("  • features_viz/week_1/")
    print("  • features_results/week_1/")
    print("="*80)

if __name__ == "__main__":
    main()
