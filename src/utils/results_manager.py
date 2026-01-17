# src/utils/results_manager.py
# =======================================
# UPDATED: Professional Results Manager
# =======================================

import os
import json
import mlflow # type: ignore
import re
from typing import Dict, Any, Optional
from mlflow.exceptions import MlflowException # type: ignore

from .utils import get_path


class ResultsManager:
    """
    Unified results manager for saving evaluation metrics and logging to MLflow.
    Works with any model type (Linear, SVR, RandomForest, XGBoost, ARIMA, LSTM, etc.).

    Responsibilities:
    - Save evaluation metrics locally as JSON
    - Log metrics and parameters to MLflow
    - Ensure directory safety and consistent structure
    """

    def __init__(self, model_type: str, week: int):
        self.model_type = model_type.lower()
        self.week = week

    # ---------------------------------------------------------
    # JSON-safe conversion for NumPy and exotic types
    # ---------------------------------------------------------
    @staticmethod
    def _json_safe(obj: Any) -> Any:
        import numpy as np

        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return obj

    # ---------------------------------------------------------
    # MLflow-safe metric name sanitization
    # ---------------------------------------------------------
    @staticmethod
    def _sanitize_metric_name(name: str) -> str:
        """
        Convert metric names into MLflow-safe names.
        - Replace % with _pct
        - Replace invalid characters with _
        - Collapse multiple underscores
        - Strip leading/trailing underscores
        """
        # Replace % explicitly
        name = name.replace("%", "_pct")

        # Replace any remaining invalid characters
        name = re.sub(r'[^a-zA-Z0-9_\-\.\/\s:]', '_', name)

        # Replace spaces with underscores
        name = name.replace(" ", "_")

        # Collapse multiple underscores
        name = re.sub(r'_+', '_', name)

        # Strip leading/trailing underscores
        name = name.strip("_")

        return name

    # ---------------------------------------------------------
    # Save results locally
    # ---------------------------------------------------------
    def save_results(self, results: Dict[str, Any], filename: str) -> str:
        """
        Save evaluation results to JSON in the corresponding results path.
        """
        path_key = f"{self.model_type}_results"

        try:
            save_dir = get_path(path_key, week=self.week)
        except ValueError:
            print(f"[WARN] Unknown model type '{self.model_type}'. Saving to general results path.")
            save_dir = get_path("results", week=self.week)

        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, filename)

        with open(save_path, "w") as f:
            json.dump(results, f, indent=2, default=self._json_safe)

        print(f"✅ Results saved locally to: {save_path}")
        return save_path

    # ---------------------------------------------------------
    # Log results to MLflow + save local copy
    # ---------------------------------------------------------
    def log_results_mlflow(
        self,
        results: Dict[str, Any],
        params: Optional[Dict[str, Any]] = None,
        filename: str = "mlflow_metrics.json",
        run_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
    ) -> str:
        """
        Log metrics and parameters to MLflow and also save them locally.
        """

        # Sanitize metric names
        sanitized_results = {
            self._sanitize_metric_name(k): v for k, v in results.items()
        }

        # Create or select experiment
        experiment_name = self.model_type.upper()
        mlflow.set_experiment(experiment_name)

        try:
            with mlflow.start_run(run_name=run_name):
                # Add tags
                if tags:
                    mlflow.set_tags(tags)

                # Log parameters
                if params:
                    mlflow.log_params(params)

                # Log metrics
                for key, value in sanitized_results.items():
                    if isinstance(value, (int, float)):
                        try:
                            mlflow.log_metric(key, float(value))
                        except MlflowException as e:
                            print(f"⚠ Failed to log metric '{key}': {e}")

        except MlflowException as e:
            print(f"⚠ Error during MLflow run: {e}")

        print(f"📊 Metrics logged to MLflow under experiment '{experiment_name}'.")

        # Save metrics locally
        try:
            save_dir = get_path("mlflow", week=self.week)
        except ValueError:
            save_dir = get_path("mlflow")

        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, filename)

        with open(save_path, "w") as f:
            json.dump(
                {"params": params, "metrics": sanitized_results, "tags": tags},
                f,
                indent=2,
                default=self._json_safe,
            )

        print(f"💾 MLflow metrics also saved locally to: {save_path}")
        return save_path
