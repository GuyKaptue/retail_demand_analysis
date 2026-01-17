# src/core/week_3/__init__.py

"""
Week 3: Advanced Machine Learning & Deep Learning Models
=========================================================

This package provides a comprehensive suite of ML and DL models for time series
regression tasks, including data preparation, model training, hyperparameter tuning,
evaluation, and visualization.

Package Components:
-------------------

1. Data Preparation (preparing/)
   - DataPreparer: Automated data preprocessing and feature engineering
   - MLConfig: Centralized configuration management
   - ModelVisualizer: Comprehensive visualization toolkit

2. Machine Learning Models (models/ml/)
   - LinearRegressionRunner: Ridge regression with L2 regularization
   - RandomForestRunner: Ensemble of decision trees
   - SVRRunner: Support Vector Regression
   - XGBoostRunner: Gradient boosted decision trees

   Each ML model supports:
   - Baseline training
   - GridSearchCV tuning
   - RandomizedSearchCV tuning
   - Hyperopt tuning (Bayesian optimization)

3. Deep Learning Models (models/deep_learning/)
   - LSTMRunner: LSTM networks for sequence modeling
   - LSTMForecaster: LSTM-based forecasting
   - Complete pipeline from data prep to evaluation
   - Custom sequence building and scaling

Core Features:
--------------
- End-to-end pipeline automation
- Multiple hyperparameter optimization strategies
- Comprehensive metrics (RMSE, MAE, R²)
- Rich visualization suite
- MLflow experiment tracking
- Modular and extensible architecture

Quick Start Examples:
---------------------

Traditional ML Workflow:
    >>> from src.core.week_3 import XGBoostRunner
    >>>
    >>> runner = XGBoostRunner(week=3)
    >>> runner.train_baseline()
    >>> y_pred, metrics = runner.evaluate()
    >>> runner.plot_actual_vs_predicted(y_pred)

Deep Learning Workflow:
    >>> from src.core.week_3.models.deep_learning.lstm import LSTMRunner
    >>>
    >>> lstm_runner = LSTMRunner(week=3)
    >>> lstm_runner.train()
    >>> results = lstm_runner.evaluate()

Batch Model Comparison:
    >>> from src.core.week_3.models.ml import MLPipeline
    >>>
    >>> pipeline = MLPipeline(week=3)
    >>> results = pipeline.run_all_models(tuning_method='baseline')
    >>> print(results)

Full Hyperparameter Tuning:
    >>> pipeline = MLPipeline(week=3)
    >>> comparison = pipeline.full_model_comparison(
    ...     models=['xgboost', 'random_forest'],
    ...     tuning_methods=['baseline', 'hyperopt']
    ... )

Data Preparation Only:
    >>> from src.core.week_3 import DataPreparer
    >>>
    >>> preparer = DataPreparer()
    >>> X_train, X_test, y_train, y_test, groups, preprocessor = \
    ...     preparer.prepare_for_model("XGBRegressor")

Configuration Management:
    >>> from src.core.week_3 import MLConfig
    >>>
    >>> config = MLConfig()
    >>> model = config.instantiate_model("xgboost")
    >>> params = config.get_baseline_params("xgboost")

Visualization:
    >>> from src.core.week_3 import ModelVisualizer
    >>>
    >>> viz = ModelVisualizer(df=df, target='sales', y_true=y_test)
    >>> viz.set_predictions(y_pred)
    >>> viz.plot_actual_vs_predicted()
    >>> viz.plot_residuals()

Advanced Usage:
---------------

Custom Hyperparameter Search:
    >>> runner = XGBoostRunner(week=3)
    >>> runner.train_hyperopt(max_evals=200)  # More thorough search
    >>> y_pred, metrics = runner.evaluate("xgboost_custom_hyperopt")

Filtered Time Series Analysis:
    >>> runner.plot_time_series_comparison(
    ...     y_pred,
    ...     store_filter=25,
    ...     item_filter=100,
    ...     n_points=500
    ... )

LSTM with Custom Configuration:
    >>> from src.core.week_3.models.deep_learning.lstm import LSTMModel, LSTMTrainer
    >>>
    >>> model = LSTMModel(
    ...     input_size=10,
    ...     hidden_size=128,
    ...     num_layers=3,
    ...     dropout=0.3
    ... )
    >>> trainer = LSTMTrainer(model, epochs=100, batch_size=64)

Dependencies:
-------------
- scikit-learn
- xgboost
- hyperopt
- mlflow
- tensorflow (for deep learning)
- pandas
- numpy
- matplotlib
- seaborn

Directory Structure:
--------------------
week_3/
├── __init__.py              # This file
├── preparing/               # Data preparation and configuration
│   ├── __init__.py
│   ├── data_preparer.py
│   ├── ml_config.py
│   └── model_visualizer.py
└── models/                  # Model implementations
    ├── __init__.py
    ├── ml/                  # Traditional ML models
    │   ├── __init__.py
    │   ├── ml_main.py
    │   ├── ml_pipeline.py
    │   ├── regressors/
    │   │   ├── __init__.py
    │   │   ├── linear_regression.py
    │   │   ├── random_forest.py
    │   │   ├── svr.py
    │   │   └── xgboost.py
    │   └── README.md
    └── deep_learning/       # Deep learning models
        ├── __init__.py
        ├── lstm/
        │   ├── __init__.py
        │   ├── lstm_data_preparer.py
        │   ├── lstm_model.py
        │   ├── lstm_visualizer.py
        │   ├── lstm_forecaster.py
        │   ├── run_lstm.py
        │   ├── sequence_builder.py
        │   └── sequence_scaler.py
        ├── lstm_main.py
        └── README.md

MLflow Integration:
-------------------
All models automatically log to MLflow:
- Metrics (RMSE, MAE, R²)
- Parameters (baseline and tuned)
- Model artifacts
- Training metadata

View results:
    >>> import mlflow
    >>> mlflow.set_experiment("week3_models")
    >>> mlflow.ui()  # Launch MLflow UI

Performance Tips:
-----------------
1. Start with baseline models to establish benchmarks
2. Use RandomizedSearchCV for initial parameter exploration
3. Use Hyperopt for final tuning (computationally expensive)
4. Monitor experiments with MLflow to avoid redundant runs
5. Use visualization tools to diagnose model issues

Troubleshooting:
----------------
- Out of memory: Reduce dataset size or use simpler models
- Slow training: Reduce max_evals, use parallel processing
- Poor performance: Check data quality, try different models
- Import errors: Ensure all dependencies are installed

For More Information:
---------------------
See individual module documentation:
- help(DataPreparer)
- help(XGBoostRunner)
- help(LSTMRunner)
- help(MLPipeline)
"""
__version__ = "1.0.0" 
# =========================
# Data Preparation
# =========================
from .preparing import (
    DataPreparer,
    MLConfig,
    ModelVisualizer,
)

# =========================
# Machine Learning Models
# =========================
from .models import (
    LinearRegressionRunner,
    RandomForestRunner,
    SVRRunner,
    XGBoostRunner,
    ModelType,
    ModelConfig,
    MLPipelineRunner,
    MultiModelComparison,
    MLPipeline,
)

# =========================
# Deep Learning (LSTM)
# =========================
from .models import (
    LSTMDataPreparer,
    LSTMModel,
    LSTMRunner,
    LSTMVisualizer,
    LSTMForecaster,
    SequenceBuilder,
    SequenceScaler,
)

__all__ = [
    # Data prep
    "DataPreparer",
    "MLConfig",
    "ModelVisualizer",

    # ML
    "LinearRegressionRunner",
    "RandomForestRunner",
    "SVRRunner",
    "XGBoostRunner",
    "ModelType",
    "ModelConfig",
    "MLPipelineRunner",
    "MultiModelComparison",
    "MLPipeline",

    # Deep Learning
    "LSTMDataPreparer",
    "SequenceBuilder",
    "SequenceScaler",
    "LSTMModel",
    "LSTMRunner",
    "LSTMForecaster",
    "LSTMVisualizer",
]


__author__ = "Guy Kaptue"
__description__ = "Week 3 – Advanced ML & Deep Learning for Time Series Regression"
__status__ = "Production"
