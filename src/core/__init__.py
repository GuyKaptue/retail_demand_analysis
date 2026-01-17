# src/core/__init__.py
"""
Core Package – Retail Demand Analysis
====================================

This package contains all core analysis modules organized by project week.
Each week builds upon the previous, forming a complete, production-grade
retail demand forecasting pipeline.

Week 1: Data Loading, Cleaning, EDA, Feature Engineering
Week 2: Classical Time Series Models (ARIMA, SARIMA, ETS, Prophet)
Week 3: Machine Learning & Deep Learning Models
"""

# =====================================================
# WEEK 1: Data Loading, EDA & Feature Engineering
# =====================================================
from .week_1 import (
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
)

# =====================================================
# WEEK 2: Time Series Models
# =====================================================
from .week_2 import (
    PreparingData,
    TSVisualization,
    ARIMAConfig,
    ARIMAGridSearch,
    StationarityAnalyzer,
    ARIMAVisualization,
    ARIMAEvaluator,
    ARIMAReport,
    ARIMAPipeline,
    SARIMAPipeline,
    SARIMAEstimator,
    SARIMAEvaluator,
    SARIMAGridSearch,
    SARIMAConfig,
    ETSConfig,
    ETSEstimator,
    ETSEvaluator,
    ETSPipeline,
    ProphetConfig,
    ProphetEstimator,
    ProphetEvaluator,
    ProphetPipeline,
    ModelResult,
    ModelComparison,
    ModelComparisonVisualizer,
)

# =====================================================
# WEEK 3: Advanced ML & Deep Learning
# =====================================================
from .week_3 import (
    # Data Preparing
    DataPreparer,
    MLConfig,
    ModelVisualizer,
    # ML Models & Pipelines
    LinearRegressionRunner,
    RandomForestRunner,
    SVRRunner,
    XGBoostRunner,
    ModelType,
    ModelConfig,
    MLPipelineRunner,
    MultiModelComparison,
    MLPipeline,

    # Deep Learning – Data & Sequences
    LSTMDataPreparer,
    SequenceScaler,
    SequenceBuilder,

    # Deep Learning – Models & Execution
    LSTMModel,
    LSTMRunner,
    LSTMVisualizer,
    LSTMForecaster,
)

__all__ = [
    # ======================
    # WEEK 1
    # ======================
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

    # ======================
    # WEEK 2
    # ======================
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

    # ======================
    # WEEK 3
    # ======================
    
    # Data Preparation
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
    "SequenceScaler",
    "SequenceBuilder",
    "LSTMModel",
    "LSTMRunner",
    "LSTMForecaster",
    "LSTMVisualizer",
]

# =====================================================
# Package Metadata
# =====================================================
__version__ = "1.0.0"
__author__ = "Guy Kaptue"
