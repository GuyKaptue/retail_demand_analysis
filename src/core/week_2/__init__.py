# src/core/week_2/__init__.py

"""
Week 2 Core Functionality
=========================

This package provides all core components used in Week 2 of the forecasting
workflow, including:

- Data preparation utilities (filtering, aggregation, calendar completion,
  conversion to Darts TimeSeries, and visualization helpers)

- ARIMA modeling tools (stationarity analysis, visualization, evaluation,
  grid search, and a full ARIMA pipeline)

- SARIMA modeling tools (seasonal ARIMA estimator, evaluator, grid search,
  configuration manager, and the complete SARIMA pipeline)

- ETS (Exponential Smoothing) modeling tools with trend + seasonality support,
  including estimator, evaluator, and full pipeline

- Prophet modeling tools (trend, seasonality, holiday effects) with a complete
  end-to-end pipeline

- Model Comparison: Cross-model comparison for ETS, Prophet, SARIMA, and ARIMA.
                    Loads existing JSON results, produces a clean comparison table,
                    and optionally logs to MLflow.

The goal of this package is to offer a unified, clean interface for all
time-series modeling, preparation, and comparison steps used throughout Week 2.
"""

# -------------------------
# Preparing module exports
# -------------------------
from .preparing import PreparingData, TSVisualization

# -------------------------
# ARIMA, SARIMA, ETS, Prophet, and Model Comparison exports
# (all exposed through the models package)
# -------------------------
from .models import (
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
)

__all__ = [
    # Preparing
    "PreparingData",
    "TSVisualization",

    # ARIMA
    "ARIMAConfig",
    "ARIMAGridSearch",
    "StationarityAnalyzer",
    "ARIMAVisualization",
    "ARIMAEvaluator",
    "ARIMAReport",
    "ARIMAPipeline",

    # SARIMA
    "SARIMAPipeline",
    "SARIMAEstimator",
    "SARIMAEvaluator",
    "SARIMAGridSearch",
    "SARIMAConfig",

    # ETS
    "ETSConfig",
    "ETSEstimator",
    "ETSEvaluator",
    "ETSPipeline",

    # Prophet
    "ProphetConfig",
    "ProphetEstimator",
    "ProphetEvaluator",
    "ProphetPipeline",

    # Model Comparison
    "ModelResult",
    "ModelComparison",
    "ModelComparisonVisualizer",
]
