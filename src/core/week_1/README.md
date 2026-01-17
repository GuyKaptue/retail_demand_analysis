# 📊 Favorita Grocery Sales Forecasting — Week 1 Pipeline

## Overview
Week 1 establishes the **data acquisition, preprocessing, and exploratory analysis pipeline** for the Favorita dataset.  
The workflow is designed to be **scalable, reproducible, and transparent**, enabling you to handle 50M+ rows safely while producing clear plots and statistics at every stage.

This week's modules prepare the dataset for downstream **time series forecasting and feature engineering**.

---

## 📂 Project Structure

```
week_1/
│── __init__.py
│── main_week1.py              # Main entry point for Week 1 pipeline
│
├── loader/
│   ├── __init__.py
│   ├── google_drive_loader.py # Download CSVs from Google Drive
│   ├── kaggle_loader.py       # Download datasets/competitions from Kaggle
│   ├── loader.py              # Centralized local CSV loader (pandas + dask)
│   ├── train_subset_processor.py # Train subset preparation pipeline
│   └── visualizer.py          # Workflow & distribution visualizations
│
└── processor/
    ├── __init__.py
    │
    ├── eda/                   # Exploratory Data Analysis
    │   ├── __init__.py
    │   ├── eda_report_generator.py
    │   ├── retail_data_cleaner.py
    │   ├── time_series_diagnostics.py
    │   └── visualization.py
    │
    └── features/              # Feature engineering & impact analysis
        ├── __init__.py
        ├── data_preparation_pipeline.py
        ├── feature_engineering.py
        ├── feature_viz.py
        └── impact_analysis.py
```

---

## 📥 Data Acquisition Options

Week 1 supports **three flexible data loading strategies**:

### Option 1: Local Files (Skip Download)
If you already have the raw CSV files locally:

```
📁 /path/to/data/csv/raw/
├── holidays_events.csv
├── items.csv
├── oil.csv
├── sample_submission.csv
├── stores.csv
├── test.csv
├── train.csv
└── transactions.csv
```

✅ **The pipeline automatically detects existing files and skips downloads.**

**Example log output:**
```
⚡ Found existing CSV files in /Volumes/Intenso/.../raw
   Skipping download/unzip.
```

### Option 2: Kaggle API Download
Download directly from Kaggle Competition using `KaggleDataLoader`:

```python
from src.week_1.loader import KaggleDataLoader

kaggle = KaggleDataLoader()

# Option A: Download + unzip in one step
kaggle.prepare("favorita-grocery-sales-forecasting", folder_name="raw")

# Option B: Step-by-step control
kaggle.download("favorita-grocery-sales-forecasting", folder_name="raw")
kaggle.unzip("favorita-grocery-sales-forecasting", folder_name="raw")
```

**Requirements:**
- Kaggle API credentials in `~/.kaggle/kaggle.json` or environment variables
- Must join the competition at: `https://www.kaggle.com/c/favorita-grocery-sales-forecasting`

### Option 3: Google Drive Download
Download from Google Drive using file IDs stored in YAML config:

```python
from src.week_1.loader import GoogleDriveLoader

gdrive = GoogleDriveLoader()

# Method 1: Load metadata files directly into memory (fast, no disk writes)
metadata = gdrive.load_metadata_from_config("gdrive_file_ids.yaml")
# Returns: {"holiday_events": df, "items": df, "oil": df, ...}

# Method 2: Download and save all files to disk
gdrive.download_config("gdrive_file_ids.yaml", folder_name="raw")

# Method 3: Download specific files only
gdrive.download_config("gdrive_file_ids.yaml", skip=["train"])
```

**YAML Config Format** (`config/gdrive_file_ids.yaml`):
```yaml
file_ids:
  holiday_events: "1ABC...XYZ"
  items: "1DEF...XYZ"
  oil: "1GHI...XYZ"
  stores: "1JKL...XYZ"
  transactions: "1MNO...XYZ"
  train: "1PQR...XYZ"
```

**Features:**
- **Direct loading:** Metadata files load directly into DataFrames without disk writes
- **Selective downloads:** Skip large files like `train.csv` when unnecessary
- **YAML-driven:** Centralized configuration for reproducibility

---

## 🚀 Workflow Steps

### 1. **Data Acquisition**
Choose your preferred loading strategy based on your environment:

| Strategy | Use Case | Speed |
|----------|----------|-------|
| **Local Files** | Files already downloaded | ⚡ Instant |
| **Kaggle API** | Official competition data | 🐢 Slow (large files) |
| **Google Drive** | Team collaboration, cloud storage | 🚀 Fast (direct loading) |

### 2. **Load Large CSVs Efficiently**
`DataLoader` provides smart loading strategies:

```python
from src.week_1.loader import DataLoader

loader = DataLoader()

# Pandas eager loading (small-medium files)
df_stores = loader.load_csv("raw", "stores.csv")

# Dask lazy loading → pandas (large files like train.csv)
df_train = loader.load_csv_dask("raw", "train.csv")

# Bulk load all CSVs in a folder
all_data = loader.load_all_csvs("raw", week=1)
```

### 3. **Prepare Train Subset**
`TrainSubsetProcessor` intelligently filters the massive 54M+ row `train.csv`:

```python
from src.week_1.loader import TrainSubsetProcessor

processor = TrainSubsetProcessor(loader=loader)

df_subset = processor.prepare_train_subset(
    region="Guayas",           # Filter by store region
    sample_size=2_000_000,     # Final sample size
    chunk_size=1_000_000,      # Memory-safe chunked reading
    top_n=3,                   # Keep top N product families
    output_name="train_subset_guayas.csv",
    plot_workflow=True         # Generate visual workflow diagram
)
```

**Processing Pipeline:**
```
stores.csv → Get Region Stores → store_ids
items.csv  → Get Top Families → family_list
train.csv  → Chunked Reading  → Filter Stores → Filter Families → Sample → Save
```

### 4. **Transform to Daily Time Series**
Aggregate transaction-level data into daily time series:

```python
# Daily sales by product family
df_daily_family = processor.transform_to_daily(
    df_subset, 
    group_by="family", 
    agg_func="sum"
)

# Daily sales by store
df_daily_store = processor.transform_to_daily(
    df_subset, 
    group_by="store_nbr", 
    agg_func="sum"
)

# Daily sales by item
df_daily_item = processor.transform_to_daily(
    df_subset, 
    group_by="item_nbr", 
    agg_func="sum"
)
```

Outputs saved to: `processed/loader_processed/week_1/`

### 5. **Exploratory Data Analysis (EDA)**
`EDAReportGenerator` orchestrates comprehensive data quality checks:

```python
from src.week_1.processor import EDAReportGenerator

eda = EDAReportGenerator(df_subset, week=1)
report = eda.run_full_eda()
```

**EDA Components:**
- ✅ Structural summary (shape, memory, date ranges)
- ✅ Data quality checks (nulls, duplicates, negative sales)
- ✅ Missing value analysis with visualizations
- ✅ Calendar gap detection and filling
- ✅ Outlier detection (IQR, Z-score, log-transform methods)
- ✅ Distribution analysis (histograms, transformations)
- ✅ Time series diagnostics (ADF test, STL decomposition, autocorrelation)
- ✅ Seasonal pattern analysis

**Optimizations for Large Datasets:**
- Pre-aggregated visualizations (10-100x speedup)
- Sampled data for distribution plots (100k rows)
- Efficient pandas operations throughout

### 6. **Feature Engineering**
`DataPreparationPipeline` creates time series forecasting features:

```python
from src.week_1.processor import DataPreparationPipeline

prep = DataPreparationPipeline(
    df=df_subset,
    holidays_df=holidays_df,
    oil_df=oil_df,
    week=1
)

# Run full pipeline
final_df = prep.run_pipeline()
```

**Feature Categories:**
- **Date Features:** Year, month, day, day of week, week of year, is_weekend
- **Lag Features:** 7, 14, 28, 365-day lags
- **Rolling Statistics:** 7, 28, 365-day rolling mean/std
- **Promotion Features:** Current + lagged promotion indicators
- **Price Features:** Price changes, ratios (if available)
- **Holiday Distance:** Days to/from nearest holiday
- **Aggregates:** Store-level and item-level sales statistics

**Impact Analysis:**
- Oil price correlation with sales
- Holiday vs non-holiday sales comparison
- Promotional lift analysis

### 7. **Memory Optimization**
Automatic dtype optimization before saving:

```python
prep.save_final_dataset("final_train_dataset.csv")
```

**Optimizations Applied:**
- Downcast integers to smallest type (int8/16/32)
- Convert floats to float32 (balance precision vs memory)
- Typical reduction: **50-70% memory savings**

---

## 📊 Visualizations Produced

### Workflow Diagrams
- `train_subset_workflow_detailed.png` - Complete processing pipeline with decision trees

### Distribution Analysis
- `distribution_{region}.png` - Store and family distributions
- `unit_sales_distribution.png` - Sales distribution histogram

### Time Series Plots
- `daily_unit_sales_by_{family|store|item}.png` - Daily aggregated sales
- `preprocessing_steps.png` - Raw → smoothed → differenced series

### EDA Visualizations
- `missing_values_report.png` - Heatmap of missing data patterns
- `missing_calendar_coverage.png` - Date completeness analysis
- `outlier_comparison_{method}.png` - Before/after outlier handling
- `total_sales_over_time.png` - Complete time series overview
- `sales_by_store.png` - Store-level trends
- `autocorrelation.png` - ACF/PACF plots

### Statistical Diagnostics
- `stationarity_checks.png` - ADF test results
- `stl_decomposition.png` - Trend, seasonal, residual components
- `autocorrelation_diagnostics.png` - Lag structure analysis

### Feature Analysis
- `correlation_heatmap.png` - Feature correlation matrix
- `oil_vs_sales.png` - Oil price impact on sales
- `holiday_vs_nonholiday_sales.png` - Holiday impact comparison

---

## ▶️ Running the Pipeline

### Full Automated Pipeline
```bash
cd src/week_1
python main_week1.py
```

This executes the complete workflow:
1. Load raw data (auto-detects existing files or downloads)
2. Prepare train subset (region filtering + sampling)
3. Transform to daily time series (family, store, item levels)
4. Run comprehensive EDA
5. Generate features and impact analysis
6. Save all outputs with optimized memory

### Step-by-Step Notebook Execution
For interactive exploration, each component can be run independently:

```python
from src.week_1.loader import DataLoader, TrainSubsetProcessor
from src.week_1.processor import EDAReportGenerator, DataPreparationPipeline

# Step 1: Load data
loader = DataLoader()
raw_data = loader.load_all_csvs("raw", week=1)

# Step 2: Process subset
processor = TrainSubsetProcessor(loader=loader)
df_subset = processor.prepare_train_subset(region="Pichincha")

# Step 3: EDA
eda = EDAReportGenerator(df_subset, week=1)
eda.run_structural_summary()
eda.run_missing_value_report()
# ... continue with individual EDA steps

# Step 4: Feature engineering
prep = DataPreparationPipeline(df_subset, week=1)
prep.add_date_features()
prep.add_advanced_lags()
# ... continue with individual feature steps
```

---

## 📁 Output Structure

```
processed/loader_processed/week_1/
├── train_subset_guayas.csv          # Filtered & sampled subset
├── train_daily_family.csv           # Daily aggregation by family
├── train_daily_store.csv            # Daily aggregation by store
└── train_daily_item.csv             # Daily aggregation by item

eda/week_1/
├── distribution_guayas.png
├── missing_values_report.png
├── outlier_comparison_logzscore.png
├── total_sales_over_time.png
├── stl_decomposition.png
└── ... (20+ visualization files)

eda_stats/week_1/
├── structural_summary.csv           # Dataset metadata
├── data_quality_report.csv          # Quality metrics
└── pipeline_stats.json              # Processing statistics

features/week_1/
├── feature_engineered.csv           # Intermediate features
└── final_train_dataset.csv          # Complete feature set (memory-optimized)

features_viz/week_1/
├── correlation_heatmap.png
├── oil_vs_sales.png
└── holiday_vs_nonholiday_sales.png

features_results/week_1/
├── sales_with_holidays.csv          # Merged holiday data
└── impact_analysis_summary.json     # Statistical summaries
```

---

## 🔧 Configuration Files

### `config/gdrive_file_ids.yaml`
```yaml
file_ids:
  holiday_events: "your_file_id_here"
  items: "your_file_id_here"
  oil: "your_file_id_here"
  stores: "your_file_id_here"
  transactions: "your_file_id_here"
  train: "your_file_id_here"
```

### Environment Variables (Optional)
```bash
# Kaggle API credentials (if not using ~/.kaggle/kaggle.json)
export KAGGLE_USERNAME="your_username"
export KAGGLE_KEY="your_api_key"
```

---

## 💡 Best Practices

### Memory Management
- Use `DataLoader.load_csv_dask()` for files > 1GB
- Enable chunked processing for train.csv (default: 1M rows/chunk)
- Memory optimization reduces final dataset size by 50-70%

### Data Quality
- Always run EDA before feature engineering
- Check calendar completeness with `check_and_plot_missing_calendar()`
- Use `logzscore` method for outlier handling (recommended for retail data)

### Reproducibility
- Set random seeds for sampling: `random_state=42`
- Save processing parameters in YAML configs
- Version control all config files

### Performance
- Pre-aggregate data for visualizations (10-100x faster)
- Use efficient pandas operations (vectorized, not loops)
- Sample large datasets for exploratory plots (100k-1M rows)

---

## ✨ Next Steps

### Week 2: Time Series Modeling
- **Baseline Models:** Naive, seasonal naive, moving average
- **Statistical Models:** ARIMA, SARIMA, Prophet
- **Machine Learning:** XGBoost, LightGBM with time series features
- **Evaluation:** RMSE, MAE, MAPE across different horizons

### Week 3: Advanced Forecasting
- **Deep Learning:** LSTM, GRU, Transformer models
- **Ensemble Methods:** Model stacking and blending
- **Hierarchical Forecasting:** Store → family → item hierarchy
- **Production Pipeline:** Model serving and monitoring

---

## 🐛 Troubleshooting

### "Kaggle API authentication failed"
**Solution:** Ensure `~/.kaggle/kaggle.json` exists with valid credentials or set `KAGGLE_USERNAME` and `KAGGLE_KEY` environment variables.

### "You have not joined the competition"
**Solution:** Visit [Kaggle Competition Page](https://www.kaggle.com/c/favorita-grocery-sales-forecasting) and click "Join Competition".

### "OSError: [Errno 28] No space left on device"
**Solution:** 
1. Use `GoogleDriveLoader.load_metadata_from_config()` to load small files into memory
2. Enable memory optimization: `prep.save_final_dataset()` automatically optimizes
3. Delete intermediate files after processing

### "MemoryError" when loading train.csv
**Solution:**
```python
# Use Dask instead of pandas
df_train = loader.load_csv_dask("raw", "train.csv")

# Or process in chunks
processor.prepare_train_subset(chunk_size=500_000)  # Reduce chunk size
```

---

## 📚 References

- [Kaggle Competition](https://www.kaggle.com/c/favorita-grocery-sales-forecasting)
- [Pandas Time Series Documentation](https://pandas.pydata.org/docs/user_guide/timeseries.html)
- [Statsmodels Time Series Analysis](https://www.statsmodels.org/stable/tsa.html)

---

## 👥 Authors

**Data Science Team** — Building modular, reproducible pipelines for large-scale retail forecasting.

---

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.