# src/core/week_3/preparing/ml_config.py
import os
from typing import Dict, Any
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from hyperopt import hp # type: ignore

import re
import numpy as np  # noqa: F401

from scipy.stats import randint, loguniform, uniform

from src.utils.utils import load_yaml, get_path


class MLConfig:
    """
    Configuration handler for ML experiments.
    Loads YAML config and provides structured access to global and model-specific settings.
    Includes automatic parsing of distribution strings for RandomizedSearchCV.
    """

    def __init__(self, config_file: str = None):
        # Default path to config/ml_models_config.yaml
        if config_file is None:
            config_file = os.path.join(get_path("config"), "ml_models_config.yaml")

        self.config: Dict[str, Any] = load_yaml(config_file)

        # Global settings
        self.global_cfg: Dict[str, Any] = self.config.get("global", {})
        self.models_cfg: Dict[str, Any] = self.config.get("models", {})

        # Dataset path resolution
        self.dataset_path: str = os.path.join(
            get_path("features", week=1), "sales_with_holidays.csv"
        )

    # ============================================================
    # Global accessors
    # ============================================================
    def target(self) -> str:
        return self.global_cfg.get("target", "unit_sales")

    def drop_columns(self) -> list:
        return self.global_cfg.get("drop_columns", [])

    def cv_strategy(self) -> str:
        return self.global_cfg.get("cv_strategy", "GroupKFold")

    def cv_groups(self) -> str:
        return self.global_cfg.get("cv_groups", None)

    def metrics(self) -> list:
        return self.global_cfg.get("metrics", ["rmse"])

    def test_split_date(self) -> str:
        return self.global_cfg.get("test_split_date", None)

    # ============================================================
    # Model accessors
    # ============================================================
    def get_model_cfg(self, model_name: str) -> Dict[str, Any]:
        if model_name not in self.models_cfg:
            raise ValueError(f"❌ Model '{model_name}' not found in config.")
        return self.models_cfg[model_name]

    def get_baseline_params(self, model_name: str) -> Dict[str, Any]:
        return self.get_model_cfg(model_name).get("baseline", {}).get("params", {})

    # ============================================================
    # Distribution Parsing for RandomizedSearchCV
    # ============================================================
    def _parse_distribution(self, value):
        """
        Convert YAML strings like 'loguniform(0.001, 100)' into scipy.stats objects.
        
        For scipy.stats.loguniform(a, b):
        - Creates a log-uniform distribution over [a, b]
        - Both a and b must be positive (a > 0, b > 0, a < b)
        """
        if not isinstance(value, str):
            return value

        # Parse loguniform for RandomizedSearchCV
        if value.startswith("loguniform"):
            match = re.search(r'loguniform\(([\d.e+-]+),\s*([\d.e+-]+)\)', value)
            if not match:
                raise ValueError(f"Invalid loguniform format: {value}. Expected 'loguniform(a, b)'")
            
            a = float(match.group(1))
            b = float(match.group(2))
            
            if a <= 0 or b <= 0:
                raise ValueError(f"loguniform requires positive values. Got a={a}, b={b}")
            if a >= b:
                raise ValueError(f"loguniform requires a < b. Got a={a}, b={b}")
            
            return loguniform(a, b)

        # Parse randint
        if value.startswith("randint"):
            match = re.search(r'randint\((\d+),\s*(\d+)\)', value)
            if not match:
                raise ValueError(f"Invalid randint format: {value}")
            
            low = int(match.group(1))
            high = int(match.group(2))
            
            if low >= high:
                raise ValueError(f"randint requires low < high. Got low={low}, high={high}")
            
            return randint(low, high)

        # Parse uniform
        if value.startswith("uniform"):
            match = re.search(r'uniform\(([\d.e+-]+),\s*([\d.e+-]+)\)', value)
            if not match:
                raise ValueError(f"Invalid uniform format: {value}")
            
            loc = float(match.group(1))
            scale = float(match.group(2)) - loc  # uniform(a, b) -> uniform(loc=a, scale=b-a)
            
            if scale <= 0:
                raise ValueError(f"uniform requires a < b. Got a={loc}, b={loc + scale}")
            
            return uniform(loc, scale)

        return value

    def _parse_param_dist(self, param_dist: dict) -> dict:
        """
        Apply distribution parsing to all values in param_dist for RandomizedSearchCV.
        """
        parsed = {}
        for k, v in param_dist.items():
            parsed[k] = self._parse_distribution(v)
        return parsed

    # ============================================================
    # Hyperopt Space Parsing
    # ============================================================
    def _parse_hyperopt_space(self, space_dict: dict) -> dict:
        """
        Parse Hyperopt space definitions from YAML strings.
        
        Handles:
        - hp.loguniform('name', low, high) -> samples from exp(uniform(low, high))
        - hp.quniform('name', low, high, q) -> quantized uniform
        - hp.uniform('name', low, high) -> uniform
        - hp.choice('name', options) -> categorical choice
        """
        
        
        parsed_space = {}
        
        for param_name, expr in space_dict.items():
            if not isinstance(expr, str):
                parsed_space[param_name] = expr
                continue
            
            # Remove extra whitespace
            expr = expr.strip()
            
            # hp.loguniform parsing
            if 'hp.loguniform' in expr:
                # Extract: hp.loguniform('name', low, high)
                match = re.search(r"hp\.loguniform\(['\"](\w+)['\"]\s*,\s*([-\d.e+]+)\s*,\s*([-\d.e+]+)\)", expr)
                if match:
                    name = match.group(1)
                    low = float(match.group(2))
                    high = float(match.group(3))
                    parsed_space[param_name] = hp.loguniform(name, low, high)
                    continue
            
            # hp.quniform parsing
            if 'hp.quniform' in expr:
                match = re.search(r"hp\.quniform\(['\"](\w+)['\"]\s*,\s*([-\d.e+]+)\s*,\s*([-\d.e+]+)\s*,\s*([-\d.e+]+)\)", expr)
                if match:
                    name = match.group(1)
                    low = float(match.group(2))
                    high = float(match.group(3))
                    q = float(match.group(4))
                    parsed_space[param_name] = hp.quniform(name, low, high, q)
                    continue
            
            # hp.uniform parsing
            if 'hp.uniform' in expr:
                match = re.search(r"hp\.uniform\(['\"](\w+)['\"]\s*,\s*([-\d.e+]+)\s*,\s*([-\d.e+]+)\)", expr)
                if match:
                    name = match.group(1)
                    low = float(match.group(2))
                    high = float(match.group(3))
                    parsed_space[param_name] = hp.uniform(name, low, high)
                    continue
            
            # hp.choice parsing
            if 'hp.choice' in expr:
                # Extract: hp.choice('name', ['a', 'b', ...]) or hp.choice('name', [1, 2, ...])
                match = re.search(r"hp\.choice\(['\"](\w+)['\"]\s*,\s*\[(.*?)\]\)", expr)
                if match:
                    name = match.group(1)
                    options_str = match.group(2)
                    
                    # Parse options (can be strings, numbers, or None)
                    options = []
                    for opt in options_str.split(','):
                        opt = opt.strip().strip("'\"")
                        if opt.lower() == 'none':
                            options.append(None)
                        elif opt.replace('.', '', 1).replace('-', '', 1).isdigit():
                            options.append(float(opt) if '.' in opt else int(opt))
                        else:
                            options.append(opt)
                    
                    parsed_space[param_name] = hp.choice(name, options)
                    continue
            
            # If no match, raise error
            raise ValueError(f"Could not parse Hyperopt expression: {expr}")
        
        return parsed_space

    def get_tuning_cfg(self, model_name: str, method: str) -> Dict[str, Any]:
        """
        Returns tuning config and automatically parses distributions.
        
        For RandomizedSearchCV: parses 'param_dist' using scipy.stats
        For Hyperopt: parses 'space' using hyperopt expressions
        """
        cfg = self.get_model_cfg(model_name).get("tuning", {}).get(method, {}).copy()

        if method == "randomsearch" and "param_dist" in cfg:
            cfg["param_dist"] = self._parse_param_dist(cfg["param_dist"])
        
        elif method == "hyperopt" and "space" in cfg:
            cfg["space"] = self._parse_hyperopt_space(cfg["space"])

        return cfg

    # ============================================================
    # Model instantiation
    # ============================================================
    def instantiate_model(self, model_name: str, tuned: bool = False, method: str = None) -> Any:
        cfg = self.get_model_cfg(model_name)

        if not tuned:
            model_type = cfg.get("baseline", {}).get("type", "")
            params = cfg.get("baseline", {}).get("params", {})
        else:
            if method is None:
                raise ValueError("Must specify tuning method when tuned=True")
            tuning_cfg = cfg.get("tuning", {}).get(method, {})
            model_type = tuning_cfg.get("type", cfg.get("baseline", {}).get("type", ""))
            params = tuning_cfg.get("params", {})

        # Map model types to constructors
        if model_type == "LinearRegression":
            return LinearRegression(**params)
        elif model_type == "Ridge":
            return Ridge(**params)
        elif model_type == "SVR":
            return SVR(**params)
        elif model_type == "RandomForestRegressor":
            return RandomForestRegressor(**params)
        elif model_type == "XGBRegressor":
            return XGBRegressor(**params)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":
    cfg = MLConfig()

    print("Dataset path:", cfg.dataset_path)
    print("Target:", cfg.target())
    print("Drop columns:", cfg.drop_columns())

    # Baseline XGBoost
    xgb_model = cfg.instantiate_model("xgboost")
    print("XGB baseline:", xgb_model)

    # Access tuning config
    xgb_grid_cfg = cfg.get_tuning_cfg("xgboost", "gridsearch")
    print("XGB GridSearch params:", xgb_grid_cfg)
