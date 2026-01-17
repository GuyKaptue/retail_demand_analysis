# src/__init__.py

"""
Retail Demand Analysis - Source Package
========================================

A comprehensive retail demand forecasting system featuring:

- Multi-week progressive analysis pipeline
- Traditional ML and Deep Learning models
- Time series analysis with ARIMA, SARIMA, ETS, Prophet, and Model Comparison
- Automated data loading, cleaning, and feature engineering
- Experiment tracking with MLflow
- Modular, reproducible forecasting workflows

Package Structure:
------------------
- utils/: Utility functions and helpers
- core/: Core analysis modules
  - week_1/: Data loading, EDA, feature engineering
  - week_2/: Time series analysis (ARIMA, SARIMA, ETS, Prophet, Model Comparison)
  - week_3/: Advanced ML and deep learning models

Quick Start:
------------
    from src import DataLoader, XGBoostRunner

    loader = DataLoader()
    train_df, test_df = loader.load_all_data()

    runner = XGBoostRunner(week=3)
    runner.train_baseline()
    y_pred, metrics = runner.evaluate()

For detailed examples, see individual module documentation.
"""

__version__ = "1.0.0"
__author__ = "Guy Kaptue"

# =====================================================
# UTILITIES
# =====================================================
from .utils import (
    get_path,
    load_yaml,
    save_model,
    load_model,
    setup_notebook,
    ResultsManager,
)

# =====================================================
# CORE MODULES (Week 1, Week 2, Week 3)
# =====================================================
from .core import (
    # --------------------------
    # WEEK 1: Data Loading, EDA & Feature Engineering
    # --------------------------
    KaggleDataLoader,
    GoogleDriveLoader,
    DataLoader,
    TrainSubsetProcessor,
    Visualizer,
    RetailDataCleaner,
    Visualization,
    TimeSeriesDiagnostics,
    EDAReportGenerator,
    FeatureViz,
    FeatureEngineering,
    ImpactAnalysis,
    DataPreparationPipeline,

    # --------------------------
    # WEEK 2: Time Series Analysis
    # --------------------------
    PreparingData,
    TSVisualization,

    # ARIMA
    ARIMAConfig,
    ARIMAGridSearch,
    StationarityAnalyzer,
    ARIMAVisualization,
    ARIMAEvaluator,
    ARIMAReport,
    ARIMAPipeline,

    # SARIMA
    SARIMAPipeline,
    SARIMAEstimator,
    SARIMAEvaluator,
    SARIMAGridSearch,
    SARIMAConfig,

    # ETS
    ETSConfig,
    ETSEstimator,
    ETSEvaluator,
    ETSPipeline,

    # Prophet
    ProphetConfig,
    ProphetEstimator,
    ProphetEvaluator,
    ProphetPipeline,

    # Model Comparison
    ModelResult,
    ModelComparison,
    ModelComparisonVisualizer,

    # --------------------------
    # WEEK 3: Advanced ML & Deep Learning
    # --------------------------
    DataPreparer,
    MLConfig,
    ModelVisualizer,

    LinearRegressionRunner,
    RandomForestRunner,
    SVRRunner,
    XGBoostRunner,
    MLPipeline,
    MLPipelineRunner, 
    ModelType, 
    ModelConfig,

    LSTMDataPreparer,
    SequenceScaler,
    SequenceBuilder,

    LSTMModel,
    LSTMRunner,
    LSTMVisualizer,
    LSTMForecaster,
)

# =====================================================
# PUBLIC API
# =====================================================
__all__ = [
    # Utilities
    "get_path",
    "load_yaml",
    "save_model",
    "load_model",
    "setup_notebook",
    "ResultsManager",

    # WEEK 1
    "KaggleDataLoader",
    "GoogleDriveLoader",
    "DataLoader",
    "TrainSubsetProcessor",
    "Visualizer",
    "RetailDataCleaner",
    "Visualization",
    "TimeSeriesDiagnostics",
    "EDAReportGenerator",
    "FeatureViz",
    "FeatureEngineering",
    "ImpactAnalysis",
    "DataPreparationPipeline",

    # WEEK 2
    "PreparingData",
    "TSVisualization",
    "ARIMAConfig",
    "ARIMAGridSearch",
    "StationarityAnalyzer",
    "ARIMAVisualization",
    "ARIMAEvaluator",
    "ARIMAReport",
    "ARIMAPipeline",
    "SARIMAPipeline",
    "SARIMAEstimator",
    "SARIMAEvaluator",
    "SARIMAGridSearch",
    "SARIMAConfig",
    "ETSConfig",
    "ETSEstimator",
    "ETSEvaluator",
    "ETSPipeline",
    "ProphetConfig",
    "ProphetEstimator",
    "ProphetEvaluator",
    "ProphetPipeline",
    "ModelResult",
    "ModelComparison",
    "ModelComparisonVisualizer",

    # WEEK 3 — ML
    "DataPreparer",
    "MLConfig",
    "ModelVisualizer",
    "LinearRegressionRunner",
    "RandomForestRunner",
    "SVRRunner",
    "XGBoostRunner",
    "MLPipeline",
    "MLPipelineRunner", 
    "ModelType", 
    "ModelConfig",

    # WEEK 3 — Deep Learning
    "LSTMDataPreparer",
    "SequenceScaler",
    "SequenceBuilder",
    "LSTMModel",
    "LSTMRunner",
    "LSTMForecaster",
    "LSTMVisualizer",
]

