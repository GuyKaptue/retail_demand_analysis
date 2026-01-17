# app/__init__.py

from app.bootstrap import *  # noqa: F403

# Import only non-circular dependencies at package level
from .utils import (
    MODEL_REGISTRY, 
    FEATURES, 
    load_registered_model,
    build_future_features
)

from .components import (
    FeatureForecastBuilder,
    ForecastStatus, 
    ForecastRecord,  
    ModelPerformanceTracker
)

# Note: These create circular imports and should be imported directly where needed:
# - ForecastEngine (from app.ui.forecast_engine)
# - ForecastUI (from app.ui.forecast_ui)
# - ForecastVisualizer (from app.utils.visualizer)
# - AutoBestModelSelector (from app.components.model_selector)
# - BatchForecaster (from app.components.batch_forecast)
# - ExecutiveDashboard (from app.components.executive_dashboard)

__all__ = [
    "MODEL_REGISTRY", 
    "FEATURES", 
    "load_registered_model",
    "build_future_features",
    
    "FeatureForecastBuilder",
    "ForecastStatus", 
    "ForecastRecord",  
    "ModelPerformanceTracker"
]