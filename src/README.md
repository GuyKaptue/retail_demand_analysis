# Retail Demand Analysis Package 📊

A comprehensive retail demand forecasting system with progressive analysis pipeline, traditional ML, and deep learning models.

## 🎯 Package Overview

```
src/
├── __init__.py              # Main package entry point
├── utils/                   # Utility functions
│   ├── __init__.py
│   ├── path_utils.py
│   ├── yaml_utils.py
│   ├── model_utils.py
│   ├── notebook_utils.py
│   └── results_manager.py
└── core/                    # Core analysis modules
    ├── __init__.py
    ├── week_1/              # Data loading, EDA, feature engineering
    ├── week_2/              # Time series analysis, ARIMA
    └── week_3/              # Advanced ML and deep learning
```

## 🚀 Installation

```bash
# Clone repository
git clone <repository-url>
cd retail-demand-analysis

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

## 📚 Quick Start Examples

### 1. Simple Import and Usage

```python
# Import from top-level package
from src import DataLoader, XGBoostRunner

# Load data
loader = DataLoader()
train_df, test_df = loader.load_all_data()

# Train model
runner = XGBoostRunner(week=3)
runner.train_baseline()
y_pred, metrics = runner.evaluate()

print(f"RMSE: {metrics['rmse']:.3f}")
```

### 2. Week-by-Week Workflow

```python
# ========================================
# WEEK 1: Data Loading and EDA
# ========================================
from src import (
    DataLoader,
    RetailDataCleaner,
    EDAReportGenerator,
    FeatureEngineering
)

# Load data
loader = DataLoader()
train_df, test_df = loader.load_all_data()

# Clean data
cleaner = RetailDataCleaner()
clean_df = cleaner.clean(train_df)

# Generate EDA report
eda = EDAReportGenerator(clean_df)
eda.generate_full_report()

# Feature engineering
fe = FeatureEngineering(clean_df)
enriched_df = fe.create_all_features()

# ========================================
# WEEK 2: Time Series Analysis
# ========================================
from src import (
    PreparingData,
    ARIMAPipeline,
    ARIMAReport
)

# Prepare time series data
prep = PreparingData()
ts_data = prep.prepare_for_arima()

# Run ARIMA analysis
pipeline = ARIMAPipeline()
results = pipeline.run_full_analysis(ts_data)

# Generate report
report = ARIMAReport()
report.generate_report(results)

# ========================================
# WEEK 3: Advanced ML Models
# ========================================
from src import XGBoostRunner, RandomForestRunner
from src.core.week_3.models.ml.ml_main import MLPipeline

# Option 1: Single model
xgb = XGBoostRunner(week=3)
xgb.train_hyperopt(max_evals=100)
y_pred, metrics = xgb.evaluate()

# Option 2: Compare all models
pipeline = MLPipeline(week=3)
comparison = pipeline.run_all_models(tuning_method='baseline')
print(comparison)
```

### 3. Complete End-to-End Pipeline

```python
from src import (
    # Week 1
    DataLoader,
    RetailDataCleaner,
    FeatureEngineering,
    
    # Week 2
    PreparingData,
    ARIMAPipeline,
    
    # Week 3
    XGBoostRunner,
    LSTMRunner,
)

# ========================================
# STEP 1: Load and Clean Data
# ========================================
print("📥 Loading data...")
loader = DataLoader()
train_df, test_df = loader.load_all_data()

cleaner = RetailDataCleaner()
clean_train = cleaner.clean(train_df)

# ========================================
# STEP 2: Feature Engineering
# ========================================
print("🔧 Engineering features...")
fe = FeatureEngineering(clean_train)
enriched_df = fe.create_all_features()

# ========================================
# STEP 3: Time Series Baseline
# ========================================
print("📈 Running ARIMA baseline...")
prep = PreparingData()
ts_data = prep.prepare_for_arima()

arima_pipeline = ARIMAPipeline()
arima_results = arima_pipeline.run_full_analysis(ts_data)

# ========================================
# STEP 4: Machine Learning Models
# ========================================
print("🤖 Training ML models...")
from src.core.week_3.models.ml.ml_main import MLPipeline

ml_pipeline = MLPipeline(week=3)
ml_results = ml_pipeline.run_all_models(tuning_method='baseline')

# Find best model
best_model = ml_results.nsmallest(1, 'rmse').iloc[0]
print(f"\n🏆 Best Model: {best_model['model']}")
print(f"   RMSE: {best_model['rmse']:.3f}")
print(f"   R²: {best_model['r2']:.3f}")

# ========================================
# STEP 5: Deep Learning (Optional)
# ========================================
print("🧠 Training LSTM...")
lstm = LSTMRunner(week=3)
lstm.train()
lstm_results = lstm.evaluate()

# ========================================
# STEP 6: Final Comparison
# ========================================
print("\n📊 Final Model Comparison:")
print(f"ARIMA RMSE: {arima_results['rmse']:.3f}")
print(f"Best ML RMSE: {best_model['rmse']:.3f}")
print(f"LSTM RMSE: {lstm_results['test_rmse']:.3f}")
```

## 🎨 Advanced Usage Patterns

### Pattern 1: Custom Data Pipeline

```python
from src import DataLoader, FeatureEngineering, DataPreparer

# Custom data loading
loader = DataLoader()
train_df = loader.load_train_data()

# Apply custom transformations
fe = FeatureEngineering(train_df)
fe.create_time_features()
fe.create_lag_features(lags=[1, 7, 14, 28])
fe.create_rolling_features(windows=[7, 14, 30])
enriched_df = fe.get_dataframe()

# Prepare for modeling with custom config
preparer = DataPreparer()
X_train, X_test, y_train, y_test, groups, preprocessor = \
    preparer.prepare_for_model("XGBRegressor")
```

### Pattern 2: Experiment Tracking

```python
import mlflow
from src import XGBoostRunner, RandomForestRunner

mlflow.set_experiment("retail_demand_comparison")

# Run multiple experiments
models = [XGBoostRunner, RandomForestRunner]
tuning_methods = ['baseline', 'hyperopt']

for ModelClass in models:
    for method in tuning_methods:
        with mlflow.start_run(run_name=f"{ModelClass.__name__}_{method}"):
            runner = ModelClass(week=3)
            
            if method == 'baseline':
                runner.train_baseline()
            else:
                runner.train_hyperopt(max_evals=50)
            
            y_pred, metrics = runner.evaluate()
            
            # Log additional metrics
            mlflow.log_param("model_class", ModelClass.__name__)
            mlflow.log_param("tuning_method", method)
```

### Pattern 3: Cross-Validation Pipeline

```python
from src import DataPreparer, XGBoostRunner
from sklearn.model_selection import TimeSeriesSplit
import numpy as np

preparer = DataPreparer()
X, y = preparer.get_features_target()

tscv = TimeSeriesSplit(n_splits=5)
results = []

for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
    print(f"\n=== Fold {fold + 1} ===")
    
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    runner = XGBoostRunner(week=3)
    runner.X_train, runner.y_train = X_train, y_train
    runner.X_test, runner.y_test = X_val, y_val
    
    runner.train_baseline()
    _, metrics = runner.evaluate()
    results.append(metrics)

print(f"\nCross-Validation Results:")
print(f"RMSE: {np.mean([r['rmse'] for r in results]):.3f} ± {np.std([r['rmse'] for r in results]):.3f}")
print(f"R²: {np.mean([r['r2'] for r in results]):.3f} ± {np.std([r['r2'] for r in results]):.3f}")
```

### Pattern 4: Ensemble Modeling

```python
from src import XGBoostRunner, RandomForestRunner, LSTMRunner
import numpy as np

# Train multiple models
print("Training XGBoost...")
xgb = XGBoostRunner(week=3)
xgb.train_hyperopt(max_evals=100)
xgb_pred, xgb_metrics = xgb.evaluate()

print("Training Random Forest...")
rf = RandomForestRunner(week=3)
rf.train_hyperopt(max_evals=80)
rf_pred, rf_metrics = rf.evaluate()

print("Training LSTM...")
lstm = LSTMRunner(week=3)
lstm.train()
lstm_results = lstm.evaluate()
lstm_pred = lstm_results['predictions']

# Simple averaging ensemble
ensemble_pred = (xgb_pred + rf_pred + lstm_pred) / 3

# Weighted ensemble (based on validation performance)
weights = [0.5, 0.3, 0.2]  # XGBoost, RF, LSTM
weighted_ensemble = (
    weights[0] * xgb_pred + 
    weights[1] * rf_pred + 
    weights[2] * lstm_pred
)

# Evaluate ensemble
from sklearn.metrics import root_mean_squared_error, r2_score

ensemble_rmse = root_mean_squared_error(xgb.y_test, ensemble_pred)
ensemble_r2 = r2_score(xgb.y_test, ensemble_pred)

print(f"\n🎯 Ensemble Results:")
print(f"Simple Average RMSE: {ensemble_rmse:.3f}")
print(f"Weighted Average R²: {ensemble_r2:.3f}")
```

## 🔧 Configuration

### Using MLConfig

```python
from src import MLConfig

config = MLConfig()

# Get model parameters
baseline_params = config.get_baseline_params("xgboost")
print(baseline_params)

# Get tuning configuration
tuning_cfg = config.get_tuning_cfg("random_forest", "hyperopt")
print(tuning_cfg)

# Instantiate models with different configurations
baseline_model = config.instantiate_model("xgboost")
tuned_model = config.instantiate_model("xgboost", tuned=True, method="hyperopt")
```

### Custom Configuration Files

```python
from src import load_yaml

# Load custom config
config = load_yaml("configs/custom_model_config.yaml")

# Use with your models
runner = XGBoostRunner(week=3)
runner.model.set_params(**config['xgboost']['params'])
```

## 📊 Visualization

### Comprehensive Visualization Suite

```python
from src import ModelVisualizer, XGBoostRunner

# Train model
runner = XGBoostRunner(week=3)
runner.train_baseline()
y_pred, metrics = runner.evaluate()

# Create visualizer
viz = ModelVisualizer(
    df=runner.prep.df,
    target='sales',
    y_true=runner.y_test,
    dates=runner.prep.X_test['date'],
    store_nbr=runner.prep.X_test['store_nbr'],
    item_nbr=runner.prep.X_test['item_nbr']
)

viz.set_predictions(y_pred)

# Generate all visualizations
viz.plot_target_distribution()
viz.plot_feature_correlation()
viz.plot_actual_vs_predicted()
viz.plot_residuals()
viz.plot_time_series_comparison(n_points=500)

# Filtered analysis
viz.plot_time_series_comparison(
    store_filter=25,
    item_filter=100,
    n_points=200
)
```

### LSTM-Specific Visualization

```python
from src import LSTMRunner, LSTMVisualizer

runner = LSTMRunner(week=3)
runner.train()
results = runner.evaluate()

# Use built-in visualization
runner.plot_predictions()
runner.plot_training_history()

# Or use standalone visualizer
viz = LSTMVisualizer(results)
viz.plot_predictions()
viz.plot_loss_curves()
```

## 🛠️ Utility Functions

### Path Management

```python
from src import get_path

# Get standard paths
data_path = get_path('data')
models_path = get_path('models', week=3)
results_path = get_path('results', week=3, model_type='xgboost')

print(data_path)      # data/
print(models_path)    # models/week_3/
print(results_path)   # results/week_3/xgboost/
```

### Model Persistence

```python
from src import save_model, load_model

# Save trained model
save_model(runner.pipeline, "xgboost", week=3, filename="xgb_best.pkl")

# Load model later
loaded_pipeline = load_model("xgboost", week=3, filename="xgb_best.pkl")
predictions = loaded_pipeline.predict(X_test)
```

### Results Management

```python
from src import ResultsManager

manager = ResultsManager(model_type="xgboost", week=3)

# Save results
metrics = {"rmse": 0.342, "mae": 0.198, "r2": 0.876}
manager.save_results(metrics, filename="xgb_hyperopt_results.json")

# Log to MLflow
manager.log_results_mlflow(
    metrics=metrics,
    params={"n_estimators": 500, "max_depth": 8},
    filename="xgb_mlflow.json"
)
```

### Notebook Setup

```python
from src import setup_notebook

# Configure notebook environment
setup_notebook()

# This sets up:
# - Matplotlib settings
# - Pandas display options
# - Warning filters
# - Jupyter widgets
```

## 📋 Complete API Reference

### Week 1 Components

```python
from src import (
    # Data Loading
    KaggleDataLoader,      # Load from Kaggle API
    GoogleDriveLoader,     # Load from Google Drive
    DataLoader,            # Unified data loader
    TrainSubsetProcessor,  # Process subsets of training data
    Visualizer,            # Basic visualization
    
    # Data Cleaning
    RetailDataCleaner,     # Clean retail data
    
    # EDA
    Visualization,         # EDA visualizations
    TimeSeriesDiagnostics, # Time series diagnostics
    EDAReportGenerator,    # Generate comprehensive EDA reports
    
    # Feature Engineering
    FeatureViz,            # Feature visualization
    FeatureEngineering,    # Create features
    ImpactAnalysis,        # Analyze feature impact
    DataPreparationPipeline, # End-to-end data prep
)
```

### Week 2 Components

```python
from src import (
    # Data Preparation
    PreparingData,         # Prepare for time series analysis
    TSVisualization,       # Time series visualization
    
    # ARIMA
    simple_arima_search,   # Quick ARIMA parameter search
    StationarityAnalyzer,  # Test for stationarity
    ARIMAVisualization,    # ARIMA-specific plots
    evaluate_arima,        # Evaluate ARIMA models
    ARIMAPipeline,         # Full ARIMA pipeline
    ARIMAReport,           # Generate ARIMA reports
)
```

### Week 3 Components

```python
from src import (
    # Data Preparation
    DataPreparer,          # ML data preparation
    MLConfig,              # Model configuration
    ModelVisualizer,       # Model visualization
    
    # ML Models
    LinearRegressionRunner, # Ridge regression
    RandomForestRunner,     # Random forest
    SVRRunner,              # Support vector regression
    XGBoostRunner,          # XGBoost
    
    # Deep Learning
    LSTMDataPreparer,      # LSTM data preparation
    SequenceScaler,        # Scale sequences
    SequenceBuilder,       # Build sequences
    LSTMModel,             # LSTM model
    LSTMTrainer,           # LSTM trainer
    LSTMEvaluator,         # LSTM evaluator
    LSTMVisualizer,        # LSTM visualization
    LSTMRunner,            # LSTM pipeline
)
```

## 🐛 Troubleshooting

### Common Import Errors

```python
# If you get ImportError, ensure package is installed
pip install -e .

# Or add to Python path
import sys
sys.path.append('/path/to/retail-demand-analysis')
```

### Memory Issues

```python
# Reduce dataset size
from src import DataLoader

loader = DataLoader()
train_df = loader.load_train_data()

# Sample data
train_sample = train_df.sample(frac=0.1, random_state=42)

# Use with models
runner = XGBoostRunner(week=3)
# ... use train_sample instead of full dataset
```

### Performance Optimization

```python
# Use parallel processing
runner = XGBoostRunner(week=3)
runner.model.set_params(n_jobs=-1)

# Reduce hyperparameter search
runner.train_hyperopt(max_evals=30)  # Instead of 120
```

## 📚 Additional Resources

- **MLflow UI**: `mlflow ui --port 5000`
- **Jupyter Notebooks**: See `notebooks/` directory for examples
- **Documentation**: Run `help(ComponentName)` for detailed docs
- **Tests**: Run `pytest tests/` for unit tests

## 🤝 Contributing

To extend the package:

1. Add new components to appropriate week directory
2. Update `__init__.py` files to export new components
3. Add documentation and examples
4. Write unit tests
5. Update this README

## 📝 License

Internal project - MIT License

## 👤 Author

Guy Kaptue

---

**Package Version**: 1.0.0  
**Last Updated**: December 2024  
**Status**: Production Ready