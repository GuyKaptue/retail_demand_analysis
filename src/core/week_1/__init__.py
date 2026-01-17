# src/week_1/__init__.py
"""
Week 1 module for retail demand analysis.
"""

from .loader import (
    KaggleDataLoader,
    GoogleDriveLoader,
    DataLoader,  
    TrainSubsetProcessor,
    Visualizer,
)
from .processor import (
    
    # EDA components
    RetailDataCleaner,
    Visualization,
    TimeSeriesDiagnostics,
    EDAReportGenerator,
    
    # Feature Engineering
    FeatureViz,
    FeatureEngineering,
    ImpactAnalysis,
    DataPreparationPipeline 
)
# from .feature_engineering import FeatureEngineering
# from .impact_analysis import ImpactAnalysis 
# from .time_series_eda import TimeSeriesEDA

__all__ = [
    # Loader components
    'KaggleDataLoader',
    'GoogleDriveLoader',
    'DataLoader',
    'TrainSubsetProcessor',
    'Visualizer',
    
    # Processor components
    
    # EDA components
    'RetailDataCleaner',
    'Visualization',
    'TimeSeriesDiagnostics',
    'EDAReportGenerator',
    
    # Feature Engineering
    'FeatureViz',
    'FeatureEngineering',
    'ImpactAnalysis',
    'DataPreparationPipeline'
]
