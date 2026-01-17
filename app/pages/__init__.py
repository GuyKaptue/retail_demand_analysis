# app/pages/__init__.py
"""
Pages package for the Retail Forecasting Platform.

This module ensures that all Streamlit pages under `app/pages/`
can import shared utilities such as:

- app.bootstrap
- app.utils
- src.get_path

No Streamlit code should run here.
"""

from app.bootstrap import *  # Ensures project root is on sys.path  # noqa: F403
from .forecast import Forecast
__all__ = [
    "Forecast"
]
