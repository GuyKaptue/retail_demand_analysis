# Week 3 Models Package 🧠

**End-to-end machine learning and deep learning pipeline for time series regression tasks with comprehensive hyperparameter tuning and experiment tracking**

**Author**: Guy Kaptue  
**Organization**: MASTERSCHOOL  
**Location**: Germany (Europe/Berlin)  
**Timezone**: UTC+01:00

---

## 📁 Package Structure

```
src/core/week_3/
├── __init__.py
├── models/                          # 🤖 Models Package
│   ├── __init__.py                  # Main models package initialization
│   ├── ml/                          # 📊 Traditional ML models
│   │   ├── __init__.py              # ML package initialization
│   │   ├── ml_main.py               # Main orchestrator and CLI
│   │   ├── ml_pipeline.py           # Pipeline for running multiple models
│   │   ├── regressors/              # Individual model implementations
│   │   │   ├── __ini__.py           # Regressor exports
│   │   │   ├── linear_regression.py # Linear Regression (Ridge) implementation
│   │   │   ├── random_forest.py     # Random Forest implementation
│   │   │   ├── svr.py               # Support Vector Regression implementation
│   │   │   └── xgboost.py           # XGBoost implementation
│   │   └── README.md                # ML models documentation
│   └── deep_learning/               # 🧠 Deep learning models
│       ├── __init__.py              # DL package initialization
│       ├── lstm_main.py             # LSTM main execution script
│       ├── lstm/                    # LSTM implementation
│       │   ├── __init__.py          # LSTM exports
│       │   ├── lstm_data_preparer.py # LSTM data preparation
│       │   ├── lstm_model.py        # LSTM model architecture
│       │   ├── lstm_forecaster.py   # LSTM forecasting utilities
│       │   ├── lstm_visualizer.py   # LSTM visualization
│       │   ├── run_lstm.py          # LSTM runner (main class)
│       │   ├── sequence_builder.py  # Sequence building utilities
│       │   └── sequence_scaler.py   # Sequence scaling utilities
│       ├── LSTM_STRUCTURE.md        # LSTM architecture documentation
│       ├── QUICKREF.md              # Quick reference guide
│       └── README.md                # Deep learning documentation
└── preparing/                       # 🔧 Data Preparation Module
    ├── __init__.py
    ├── data_preparer.py             # Core data preparation class
    ├── ml_config.py                 # Configuration management
    └── model_visualizer.py          # Visualization utilities
```

---

## 🖥️ Environment Configuration

### Machine Learning Models
**Conda Environment**: `retail_env`

```bash
# Create environment
conda create -n retail_env python=3.10 -y
conda activate retail_env

# Install dependencies
pip install scikit-learn xgboost hyperopt mlflow pandas numpy matplotlib seaborn
```

### Deep Learning Models
**Standard Python Environment**:`tf_env_310` (no Conda required)

```bash
# Create environement 
python3.10 -m venv tf_env_310
source tf_env_310/bin/activate

# For Apple Silicon users (M1/M2/M3):
pip install tensorflow-macos tensorflow-metal

# For Intel/AMD users:
pip install tensorflow

# Common dependencies
pip install numpy pandas scikit-learn mlflow matplotlib seaborn ipython jupyter
```

### Environment Activation

```bash
# For ML models (Conda)
conda activate retail_env

# For LSTM models (Standard Python - no activation needed)
python --version  # Verify Python 3.10+
```

### Verification

```python
# For ML models
import sklearn, xgboost, hyperopt, mlflow
print(f"scikit-learn: {sklearn.__version__}")
print(f"xgboost: {xgboost.__version__}")
print(f"hyperopt: {hyperopt.__version__}")

# For LSTM models
import tensorflow as tf
print(f"TensorFlow: {tf.__version__}")
print(f"GPU Available: {tf.config.list_physical_devices('GPU')}")
# On Apple Silicon, should show: PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')
```

---

## 🎯 Available Models

### Machine Learning Models (ML)

| Model | Class | File | Description | Output Path |
|-------|-------|------|-------------|-------------|
| **Linear Regression** | `LinearRegressionRunner` | `linear_regression.py` | Ridge regression with L2 regularization | `reports/results/week_3/linear/` |
| **Random Forest** | `RandomForestRunner` | `random_forest.py` | Ensemble of decision trees | `reports/results/week_3/random_forest/` |
| **SVR** | `SVRRunner` | `svr.py` | Support Vector Regression with RBF/linear kernels | `reports/results/week_3/svr/` |
| **XGBoost** | `XGBoostRunner` | `xgboost.py` | Gradient boosted decision trees | `reports/results/week_3/xgboost/` |

### Deep Learning Models (DL)

| Model | Class | File | Description | Output Path |
|-------|-------|------|-------------|-------------|
| **LSTM** | `LSTMRunner` | `run_lstm.py` | LSTM networks for sequence modeling | `reports/results/week_3/lstm/` |

**Note**: `lstm_main.py` provides execution scripts for both notebook and script usage of `LSTMRunner`.

---

## 🔧 Features

### Training Methods
Each model supports **4 training approaches**:
1. **Baseline** - Quick training with default parameters
2. **GridSearchCV** - Exhaustive search over parameter grid
3. **RandomizedSearchCV** - Random sampling from parameter distributions
4. **Hyperopt** - Bayesian optimization with Tree-structured Parzen Estimator (TPE)

### Evaluation Metrics
- **RMSE** (Root Mean Squared Error) - Lower is better
- **MAE** (Mean Absolute Error) - Lower is better
- **R²** (Coefficient of Determination) - Higher is better (0.7-0.95 is good)
- **MAPE** (Mean Absolute Percentage Error) - <15% is good
- **sMAPE** (Symmetric MAPE) - <20% is good

### Visualization Capabilities
- **EDA Plots**: Target distribution, feature correlations, time series analysis
- **Model Performance**: Actual vs Predicted, residual analysis, time series comparison
- **Feature Analysis**: Feature importance, coefficients, support vectors
- **Forecasting**: Future predictions with uncertainty bands, distribution plots

### Experiment Tracking
- **MLflow** integration for all experiments
- Automatic model versioning and artifact storage
- Comprehensive metric and parameter logging
- Experiment comparison and visualization

---

## 🚀 Quick Start

### Machine Learning Models

#### Basic Usage (Single Model)

```python
from src.core.week_3.models.ml.regressors.linear_regression import LinearRegressionRunner

# Initialize and train
runner = LinearRegressionRunner(week=3)
runner.train_baseline()

# Evaluate
y_pred, metrics = runner.evaluate(model_name="linear_regression")
print(f"RMSE: {metrics['rmse']:.3f}, R²: {metrics['r2']:.3f}")

# Visualize
runner.plot_actual_vs_predicted(y_pred)
runner.plot_residuals(y_pred)
```

#### Using the ML Pipeline Orchestrator

```python
from src.core.week_3.models.ml.ml_main import MLPipeline

# Initialize pipeline
pipeline = MLPipeline(week=3)

# Run single model with specific tuning
result = pipeline.run_model(
    model_name='xgboost',
    tuning_method='hyperopt',
    visualize=True,
    max_evals=100
)

# Run all models with baseline
df_results = pipeline.run_all_models(tuning_method='baseline', visualize=True)

# Full comparison (all models × all tuning methods)
df_comparison = pipeline.full_model_comparison()
```

#### Command Line Interface (ML)

```bash
# Navigate to the ml directory
cd src/core/week_3/models/ml

# Run single model with baseline
python ml_main.py --model linear --tuning baseline --visualize

# Run all models with GridSearch
python ml_main.py --all --tuning gridsearch

# Full comparison across all models and tuning methods
python ml_main.py --full-comparison

# Run specific model with Hyperopt
python ml_main.py --model random_forest --tuning hyperopt --max-evals 100 --week 3
```

### Deep Learning Models (LSTM)

#### Basic Univariate LSTM

```python
from src.core.week_3.models.deep_learning.lstm import LSTMRunner

# Initialize runner
runner = LSTMRunner(
    week=3,
    seq_len=30,                    # 30-day sequence length
    target="unit_sales",
    scaling_method="minmax"        # MinMax scaling for univariate
)

# Run full pipeline
results = runner.run_full_pipeline(
    lstm_units=[128, 64],          # Two-layer LSTM architecture
    epochs=50,
    batch_size=32,
    visualize=True,
    save_model_after=True
)
```

#### Multivariate LSTM with Covariates

```python
# Initialize with covariates
runner = LSTMRunner(
    week=3,
    seq_len=30,
    target="unit_sales",
    scaling_method="standard",     # Standard scaling for multivariate
    covariates=['onpromotion', 'dcoilwtico', 'is_holiday']
)

# Run with deeper architecture
results = runner.run_full_pipeline(
    lstm_units=[256, 128, 64],     # Three-layer LSTM
    dropout_rate=0.3,
    recurrent_dropout=0.2,
    epochs=100,
    batch_size=128,
    visualize=True
)
```

#### Step-by-Step LSTM Execution (from lstm_main.py)

```python
from src.core.week_3.models.deep_learning.lstm_main import run_step_by_step_univariate

# Execute full step-by-step pipeline
runner, metrics = run_step_by_step_univariate()

# Or use individual methods:
from src.core.week_3.models.deep_learning.lstm import LSTMRunner

runner = LSTMRunner(week=3, seq_len=30)

# Step 1: Prepare data
X_train, y_train, X_test, y_test = runner.prepare_data()
runner.display_data_summary()

# Step 2: Build model
runner.build_model(lstm_units=[128, 64], dropout_rate=0.2)

# Step 3: Train
runner.train(epochs=50, batch_size=32)
runner.plot_training_history()

# Step 4: Evaluate
y_pred, metrics = runner.evaluate()
runner.display_metrics()

# Step 5: Visualize
runner.visualize_all(n_points=200)

# Step 6: Forecast
forecast_results = runner.forecast_future(forecast_horizon=30)
```

---

## 📊 CLI Options

### ML Pipeline CLI

| Option | Description | Default | Example |
|--------|-------------|---------|---------|
| `--model` | Model to run: `linear`, `random_forest`, `svr`, `xgboost` | - | `--model xgboost` |
| `--tuning` | Tuning method: `baseline`, `gridsearch`, `randomsearch`, `hyperopt` | `baseline` | `--tuning hyperopt` |
| `--all` | Run all models with specified tuning method | `False` | `--all` |
| `--full-comparison` | Run comprehensive comparison | `False` | `--full-comparison` |
| `--visualize` | Generate visualizations | `False` | `--visualize` |
| `--max-evals` | Max evaluations for Hyperopt | Model-specific | `--max-evals 100` |
| `--week` | Week number | `3` | `--week 3` |

### LSTM Execution Methods (via lstm_main.py)

The `lstm_main.py` script provides multiple execution methods:

```python
# Method 1: Step-by-step univariate
from src.core.week_3.models.deep_learning.lstm_main import run_step_by_step_univariate
runner, metrics = run_step_by_step_univariate()

# Method 2: Step-by-step multivariate
from src.core.week_3.models.deep_learning.lstm_main import run_step_by_step_multivariate
runner, metrics = run_step_by_step_multivariate()

# Method 3: Full pipeline basic
from src.core.week_3.models.deep_learning.lstm_main import run_full_pipeline_basic
runner, results = run_full_pipeline_basic()

# Method 4: Model comparison
from src.core.week_3.models.deep_learning.lstm_main import run_model_comparison
comparison_df = run_model_comparison()

# Method 5: Load and predict
from src.core.week_3.models.deep_learning.lstm_main import run_load_and_predict
runner, metrics = run_load_and_predict("lstm_univariate_seq30_20240115_143022")

# Method 6: Hyperparameter tuning
from src.core.week_3.models.deep_learning.lstm_main import run_hyperparameter_tuning
tuning_df = run_hyperparameter_tuning()
```

---

## 🎨 Example Workflows

### Workflow 1: Quick Baseline Comparison (ML)

```python
from src.core.week_3.models.ml.ml_main import MLPipeline

# Run all ML models with baseline parameters
pipeline = MLPipeline(week=3)
results = pipeline.run_all_models(tuning_method='baseline', visualize=True)

# View results
print(results)
```

**Output Structure:**
```
reports/
└── results/
    └── week_3/
        ├── linear/
        │   ├── linear_baseline_results.json
        │   └── linear_baseline.pkl
        ├── random_forest/
        │   ├── random_forest_baseline_results.json
        │   └── random_forest_baseline.pkl
        ├── svr/
        │   ├── svr_baseline_results.json
        │   └── svr_baseline.pkl
        └── xgboost/
            ├── xgboost_baseline_results.json
            └── xgboost_baseline.pkl
```

### Workflow 2: Hyperparameter Tuning for Best Model (ML)

```python
from src.core.week_3.models.ml.regressors.xgboost import XGBoostRunner

runner = XGBoostRunner(week=3)

# Try different tuning methods
runner.train_baseline()        # Quick baseline
runner.train_gridsearch()      # Exhaustive search
runner.train_randomsearch()    # Random sampling
runner.train_hyperopt(max_evals=150)  # Bayesian optimization

# Compare results
runner.plot_model_comparison()
```

**Output Structure:**
```
reports/
└── results/
    └── week_3/
        └── xgboost/
            ├── xgboost_baseline.pkl
            ├── xgboost_gridsearch.pkl
            ├── xgboost_randomsearch.pkl
            ├── xgboost_hyperopt.pkl
            ├── xgboost_hyperopt_trials.pkl
            └── xgboost_comparison.png
```

### Workflow 3: LSTM Forecasting with Full Pipeline

```python
from src.core.week_3.models.deep_learning.lstm import LSTMRunner

# Initialize and train univariate LSTM
lstm_runner = LSTMRunner(
    week=3,
    seq_len=30,
    target="unit_sales",
    scaling_method="minmax"
)

# Run full pipeline
results = lstm_runner.run_full_pipeline(
    lstm_units=[128, 64, 32],
    dropout_rate=0.2,
    epochs=100,
    batch_size=64,
    visualize=True,
    save_model_after=True
)

# Generate future forecasts
forecast_results = lstm_runner.forecast_future(
    forecast_horizon=30,
    confidence_level=0.95,
    n_simulations=100
)

# Visualize forecasts
lstm_runner.plot_forecasts(historical_points=100, show_uncertainty=True)
lstm_runner.plot_forecast_distribution()
```

**Output Structure:**
```
reports/
└── results/
    └── week_3/
        └── lstm/
            ├── lstm_univariate_seq30_20240115_*.json
            ├── lstm_univariate_seq30_20240115_*.pkl
            ├── lstm_univariate_seq30_20240115_*_scaler.pkl
            ├── lstm_univariate_seq30_20240115_*_metrics.pkl
            ├── lstm_univariate_seq30_20240115_*_predictions.csv
            └── lstm_univariate_seq30_20240115_*_forecast_30periods.csv
```

### Workflow 4: LSTM Model Comparison

```python
from src.core.week_3.models.deep_learning.lstm_main import run_model_comparison

# Compare multiple LSTM configurations
comparison_df = run_model_comparison()

# Results include:
# - Baseline (seq=30)
# - Deep architecture (seq=30)
# - Long sequence (seq=60)
# - Different scaling (standard vs minmax)

print(comparison_df.head())
```

---

## 🔍 Hyperparameter Search Spaces

### Machine Learning Models

#### Linear Regression (Ridge)
```python
{
    'alpha': (1e-7, 1e7),           # log-uniform
    'solver': ['auto', 'lbfgs', 'sag']
}
```

#### Random Forest
```python
{
    'n_estimators': (200, 1000),
    'max_depth': [6, 8, 10, 12, 20, None],
    'min_samples_split': (2, 10),
    'min_samples_leaf': (1, 5),
    'max_features': ['sqrt', 'log2', 0.5, 0.7]
}
```

#### SVR
```python
{
    'C': (1e-4, 1e4),               # log-uniform
    'epsilon': (1e-7, 1),           # log-uniform
    'gamma': ['scale', 'auto'],
    'kernel': ['rbf', 'linear']
}
```

#### XGBoost
```python
{
    'n_estimators': (300, 1200),
    'learning_rate': (1e-3, 0.2),   # log-uniform
    'max_depth': (4, 12),
    'subsample': (0.6, 1.0),
    'colsample_bytree': (0.6, 1.0),
    'reg_lambda': (1e-3, 10),       # log-uniform (L2)
    'reg_alpha': (1e-3, 2)          # log-uniform (L1)
}
```

### Deep Learning Models

#### LSTM
```python
{
    'lstm_units': [[128, 64], [256, 128, 64], [128, 64, 32]],
    'dense_units': (32, 128),
    'dropout_rate': (0.1, 0.5),
    'recurrent_dropout': (0.0, 0.3),
    'learning_rate': (1e-4, 1e-2),  # log-uniform
    'batch_size': [16, 32, 64, 128],
    'epochs': (50, 200),
    'seq_len': (15, 90),
    'l2_reg': (1e-7, 1e-4)
}
```

---

## 📈 Performance Tips

### General Tips
1. **Start with Baseline** - Always run baseline models first to establish benchmarks
2. **Use RandomSearch for Exploration** - Faster than GridSearch, good for initial parameter discovery
3. **Use Hyperopt for Final Tuning** - Best results but computationally expensive
4. **Monitor MLflow** - Track experiments to avoid redundant runs and compare configurations
5. **Adjust max_evals Based on Budget** - More evaluations = better results but longer runtime

### Model-Specific Recommendations

#### Machine Learning Models

| Model | Recommended max_evals | Typical Runtime | Best For |
|-------|----------------------|-----------------|----------|
| **Linear** | 50 | 5-10 min | Quick baseline, interpretability |
| **Random Forest** | 80 | 30-60 min | Feature importance, non-linear patterns |
| **SVR** | 80 | 40-80 min | Non-linear patterns, small datasets |
| **XGBoost** | 120 | 60-120 min | Best overall performance, handles missing data |

#### Deep Learning Models (LSTM)

**Training Tips:**
- **Start Small**: Begin with `lstm_units=[64, 32]` and `epochs=30` for quick iteration
- **Sequence Length**: Try `seq_len=30` (1 month) as baseline, adjust based on patterns
- **Batch Size**:
  - Apple Silicon: 16-32 for better GPU utilization
  - CPU: 64-128 for better performance
- **Scaling Method**:
  - Univariate: Use `minmax` scaling
  - Multivariate: Use `standard` scaling

**Performance Optimization:**
```python
# Quick iteration (development)
runner = LSTMRunner(seq_len=15)
results = runner.run_full_pipeline(
    lstm_units=[64, 32],
    epochs=30,
    batch_size=32
)

# Production training (best performance)
runner = LSTMRunner(seq_len=30)
results = runner.run_full_pipeline(
    lstm_units=[256, 128, 64],
    dropout_rate=0.3,
    recurrent_dropout=0.2,
    epochs=150,
    batch_size=64,
    patience=15
)
```

---

## 🗂️ Output Structure

All model outputs follow a standardized directory structure:

```
reports/
├── results/                        # 📁 All model results
│   └── week_3/
│       ├── linear/                 # Linear Regression outputs
│       │   ├── linear_baseline_results.json
│       │   ├── linear_baseline.pkl
│       │   ├── linear_gridsearch_results.json
│       │   ├── linear_gridsearch.pkl
│       │   ├── linear_randomsearch_results.json
│       │   ├── linear_randomsearch.pkl
│       │   ├── linear_hyperopt_results.json
│       │   ├── linear_hyperopt.pkl
│       │   └── linear_hyperopt_trials.pkl
│       ├── random_forest/          # Random Forest outputs
│       ├── svr/                    # SVR outputs
│       ├── xgboost/                # XGBoost outputs
│       ├── lstm/                   # LSTM outputs
│       │   ├── lstm_*.json         # Model metadata
│       │   ├── lstm_*.pkl          # Trained model
│       │   ├── lstm_*_scaler.pkl   # Scaler object
│       │   ├── lstm_*_metrics.pkl  # Evaluation metrics
│       │   ├── lstm_*_predictions.csv  # Test predictions
│       │   └── lstm_*_forecast_*.csv   # Future forecasts
│       └── comparisons/            # Cross-model comparisons
│           └── multi_model_comparison_*.csv
├── visualizations/                 # 📊 All visualizations
│   └── week_3/
│       ├── linear/
│       │   ├── actual_vs_predicted.png
│       │   ├── residuals.png
│       │   ├── time_series_comparison.png
│       │   └── model_comparison.png
│       ├── random_forest/
│       ├── svr/
│       ├── xgboost/
│       └── lstm/
│           ├── training_history.png
│           ├── actual_vs_predicted.png
│           ├── residuals.png
│           ├── time_series_comparison.png
│           ├── forecast_visualization.png
│           └── forecast_distribution.png
└── models/                         # 💾 Saved models
    └── week_3/
        ├── linear/
        ├── random_forest/
        ├── svr/
        ├── xgboost/
        └── lstm/
```

---

## 🔌 Integration with Existing Pipeline

### Data Preparation Module

All models integrate seamlessly with the `preparing/` module:

```python
from src.core.week_3.preparing import DataPreparer, MLConfig, ModelVisualizer

# 1. DataPreparer - Automated data preprocessing
config = MLConfig(week=3)
data_preparer = DataPreparer(config)
X_train, X_test, y_train, y_test = data_preparer.prepare_data()

# 2. Use prepared data with any model
from src.core.week_3.models.ml.regressors.xgboost import XGBoostRunner

runner = XGBoostRunner(week=3)
runner.train_baseline()

# 3. ModelVisualizer - Consistent visualization
visualizer = ModelVisualizer(week=3)
visualizer.plot_actual_vs_predicted(y_test, y_pred)
```

### Integration Benefits

- **Unified Configuration**: Centralized settings via `MLConfig`
- **Consistent Preprocessing**: All models use the same data pipeline
- **Standardized Visualization**: Common plotting interface
- **Automatic MLflow Logging**: Integrated experiment tracking
- **Modular Architecture**: Easy to extend with new models

---

## 🐛 Troubleshooting

### Common Issues and Solutions

#### Machine Learning Models

| Issue | Cause | Solution |
|-------|-------|----------|
| **Out of Memory** | Large dataset or model | Reduce dataset size, use Linear/SVR, reduce `n_jobs` |
| **Slow Training** | Too many Hyperopt evaluations | Reduce `max_evals`, use RandomSearch first |
| **Poor Performance** | Wrong hyperparameters or data quality | Check data with visualizations, try different tuning |
| **Import Errors** | Wrong environment | Activate Conda: `conda activate retail_env` |
| **MLflow Connection Error** | MLflow not running | Start MLflow: `mlflow ui` |

#### Deep Learning Models (LSTM)

| Issue | Cause | Solution |
|-------|-------|----------|
| **Out of Memory** | Batch size or sequence length too large | Reduce `batch_size` (try 16-32), reduce `seq_len` (try 15-20) |
| **Slow Training** | Too many epochs or large architecture | Reduce epochs, use smaller layers `[64, 32]` |
| **NaN Loss** | Learning rate too high or data issues | Reduce `learning_rate` to 1e-4, check for NaN/Inf in data |
| **Overfitting** | Model too complex | Increase `dropout_rate` (0.3-0.5), add L2 regularization |
| **Underfitting** | Model too simple | Increase model depth, add more layers, train longer |
| **GPU Not Detected** | TensorFlow not installed correctly | For Apple Silicon: `pip install tensorflow-macos tensorflow-metal` |

### Debug Tips

```python
# Enable verbose logging
runner = LinearRegressionRunner(week=3, verbose=True)

# Check data quality
runner.visualizer.generate_all_eda_plots()

# Validate data splits
print(f"Train: {len(runner.X_train)}, Test: {len(runner.X_test)}")

# Check MLflow runs
import mlflow
mlflow.search_runs(experiment_names=["week3_models"])
```

---

## 📝 Best Practices

### Environment Management

```bash
# ML Models - Always use Conda environment
conda activate retail_env
python ml_main.py --model xgboost --tuning hyperopt

# LSTM Models - Standard Python environment
python lstm_main.py  # No conda activation needed

# Verify environment
python -c "import tensorflow; print(tensorflow.__version__)"
```

### Model Development Workflow

1. **Start Simple**
   ```python
   # Begin with baseline to establish benchmarks
   runner.train_baseline()
   ```

2. **Check Data Quality**
   ```python
   # Always visualize before training
   runner.visualizer.generate_all_eda_plots()
   ```

3. **Experiment Tracking**
   ```python
   # Use descriptive MLflow experiment names
   mlflow.set_experiment("week3_xgboost_tuning_v2")
   ```

4. **Incremental Tuning**
   ```python
   # Start with RandomSearch, then Hyperopt
   runner.train_randomsearch()  # Quick exploration
   runner.train_hyperopt(max_evals=100)  # Fine-tuning
   ```

5. **Validation Strategy**
   ```python
   # Always use time-based splits (no shuffle)
   # Reserve 10-20% for validation
   # Test on future dates only
   ```

### Feature Engineering

```python
# Start with basic features
runner = LSTMRunner(
    week=3,
    seq_len=30,
    target="unit_sales",
    covariates=None  # Univariate first
)

# Add features incrementally
runner = LSTMRunner(
    week=3,
    seq_len=30,
    target="unit_sales",
    covariates=['onpromotion']  # Add one feature
)

# Test feature importance
runner = LSTMRunner(
    week=3,
    seq_len=30,
    target="unit_sales",
    covariates=['onpromotion', 'dcoilwtico', 'is_holiday']  # Full feature set
)
```

### Performance Optimization

```python
# For ML models: Use parallel processing
runner = XGBoostRunner(week=3)
runner.train_baseline()  # Uses n_jobs=-1 internally

# For LSTM: Adjust batch size based on hardware
# Apple Silicon (M1/M2/M3)
runner = LSTMRunner(week=3)
results = runner.run_full_pipeline(batch_size=32)

# CPU
runner = LSTMRunner(week=3)
results = runner.run_full_pipeline(batch_size=128)
```

---

## 📚 Dependencies

### Machine Learning Models (Conda: retail_env)

```python
# Core
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0

# Machine Learning
xgboost>=1.5.0
hyperopt>=0.2.5

# Experiment Tracking
mlflow>=2.0.0

# Visualization
matplotlib>=3.5.0
seaborn>=0.11.0

# Utilities
ipython>=7.30.0
jupyter>=1.0.0
```

### Deep Learning Models (Standard Python)

```python
# Core
numpy>=1.21.0
pandas>=1.3.0

# Deep Learning
tensorflow>=2.10.0               # Standard installation
tensorflow-macos>=2.10.0         # Apple Silicon only
tensorflow-metal>=0.5.0          # Apple Silicon GPU support

# ML Utilities
scikit-learn>=1.0.0
mlflow>=2.0.0

# Visualization
matplotlib>=3.5.0
seaborn>=0.11.0

# Notebook
ipython>=7.30.0
jupyter>=1.0.0
```

### Installation Commands

```bash
# ML Models (Conda)
conda create -n retail_env python=3.10 -y
conda activate retail_env
pip install numpy pandas scikit-learn xgboost hyperopt mlflow matplotlib seaborn ipython jupyter

# LSTM Models (Standard Python)
# For Apple Silicon:
pip install numpy pandas scikit-learn mlflow matplotlib seaborn ipython jupyter
pip install tensorflow-macos tensorflow-metal

# For Intel/AMD:
pip install numpy pandas scikit-learn mlflow matplotlib seaborn ipython jupyter
pip install tensorflow
```

---

## 📚 References

### Academic Papers
- [LSTM Networks](https://www.bioinf.jku.at/publications/older/2604.pdf) - Hochreiter & Schmidhuber, 1997
- [XGBoost](https://arxiv.org/abs/1603.02754) - Chen & Guestrin, 2016
- [Random Forests](https://www.stat.berkeley.edu/~breiman/randomforest2001.pdf) - Breiman, 2001

### Technical Documentation
- [scikit-learn Documentation](https://scikit-learn.org/stable/documentation.html)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [TensorFlow LSTM Guide](https://www.tensorflow.org/guide/keras/rnn)
- [Hyperopt Documentation](http://hyperopt.github.io/hyperopt/)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)

### Books and Resources
- [Forecasting: Principles and Practice](https://otexts.com/fpp3/) - Hyndman & Athanasopoulos
- [Hands-On Machine Learning](https://www.oreilly.com/library/view/hands-on-machine-learning/9781492032632/) - Aurélien Géron
- [Deep Learning](https://www.deeplearningbook.org/) - Goodfellow, Bengio, Courville

---

## 🤝 Contributing

### Adding New ML Models

1. **Create Model File**
   ```bash
   cd src/core/week_3/models/ml/regressors/
   touch gradient_boosting.py
   ```

2. **Implement Required Methods**
   ```python
   class GradientBoostingRunner:
       def train_baseline(self):
           # Implement baseline training
           pass
       
       def train_gridsearch(self):
           # Implement GridSearchCV
           pass
       
       def train_randomsearch(self):
           # Implement RandomizedSearchCV
           pass
       
       def train_hyperopt(self, max_evals=100):
           # Implement Hyperopt
           pass
   ```

3. **Update Exports**
   ```python
   # In regressors/__ini__.py
   from .gradient_boosting import GradientBoostingRunner
   __all__ = [..., 'GradientBoostingRunner']
   ```

4. **Update ML Pipeline**
   ```python
   # In ml_main.py
   MODELS = {
       ...,
       'gradient_boosting': GradientBoostingRunner
   }
   ```

5. **Document Hyperparameters**
   - Update this README with search space
   - Add performance tips
   - Document output structure

### Adding New LSTM Variants

1. **Extend LSTMRunner**
   ```python
   # In run_lstm.py
   def build_bidirectional_model(self, ...):
       # Implement bidirectional LSTM
       pass
   ```

2. **Add to lstm_main.py**
   ```python
   def run_bidirectional_lstm():
       runner = LSTMRunner(...)
       runner.build_bidirectional_model(...)
       return runner.run_full_pipeline(...)
   ```

3. **Update Documentation**
   - Add example workflow
   - Document hyperparameters
   - Add performance comparison

---

## 📄 License

This module is part of the Retail Demand Analysis project and is licensed under the MIT License.

---

## 👤 Contact & Support

**Author**: Guy Kaptue  
**Organization**: MASTERSCHOOL  
**Location**: Hamburg, Germany (Europe/Berlin)

For issues, questions, or contributions:
1. Check this README first
2. Review existing MLflow experiments
3. Check the model-specific README files in subdirectories

---

**Last Updated**: January 14, 2026  
**Version**: 2.0.0  
**Status**: Production Ready ✅

---

## 🎓 Learning Resources

### For Beginners
1. Start with Linear Regression baseline
2. Experiment with different tuning methods
3. Compare results using MLflow UI
4. Try LSTM with univariate data first

### For Advanced Users
1. Implement custom hyperparameter spaces
2. Create ensemble models
3. Develop new LSTM architectures
4. Build automated model selection pipelines

