# src/utils/utils.py
import os
import yaml  # type: ignore
import pickle # 
from typing import Dict, Any

# =======================================
# Project Root Setup
# =======================================

# Absolute path to this file
current_file = os.path.abspath(__file__)

# Project root = 3 levels above (src/utils/utils.py → src/utils → src → retail_demand_analysis)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))

# External data root (mounted disk) 
external_data_root = "/Volumes/Intenso/my_work_spaces/retail_data/corporación_favorita_grocery_dataset"

# =======================================
# Directory Paths
# =======================================

data_path = os.path.join(external_data_root, "data")
csv_path = os.path.join(data_path, "csv")
raw_path = os.path.join(csv_path, "raw")
processed_path = os.path.join(csv_path, "processed")


reports_path = os.path.join(project_root, "reports")
visualizations_path = os.path.join(reports_path, "visualizations")
results_path = os.path.join(reports_path, "results")
mlflow_path = os.path.join(reports_path, "mlflow")

config_path = os.path.join(project_root, "config")

# Dynamic week-based paths
def week_path(base: str, week: int, subfolder: str = "") -> str:
    """
    Generate week-specific paths dynamically.

    Args:
        base (str): Base directory (processed, visualizations, results).
        week (int): Week number (1–4).
        subfolder (str): Optional subfolder inside the week directory.

    Returns:
        str: Full path to the requested week directory.
    """
    path = os.path.join(base, f"week_{week}")
    if subfolder:
        path = os.path.join(path, subfolder)
    return path


# Example specialized paths
features_path = week_path(processed_path, 1, "features")
filtered_path = week_path(processed_path, 1, "filtered")
loader_processed_path = week_path(processed_path, 1, "loader_processed")
cleaned_path = week_path(processed_path, 1, "cleaned")
eda_reports_path = week_path(visualizations_path, 1, "eda")
eda_results_path = week_path(results_path, 1, "eda_stats")
loader_path = week_path(visualizations_path, 1, "loader")
feature_results_path = week_path(results_path, 1, "features_results")
feature_viz_path = week_path(visualizations_path, 1, "features_viz")

models_path = week_path(reports_path, "models")

# Time Series Models Paths
time_series_path = os.path.join(week_path(processed_path, 2, "time_series"))
arima_models_path = os.path.join(week_path(models_path, 2, "arima"))
sarima_models_path = os.path.join(week_path(models_path, 2, "sarima"))
ets_models_path = os.path.join(week_path(models_path, 2, "ets"))
prophet_models_path = os.path.join(week_path(models_path, 2, "prophet"))

# Visualizations Paths
arima_viz_path = os.path.join(week_path(visualizations_path, 2, "arima"))
sarima_viz_path = os.path.join(week_path(visualizations_path, 2, "sarima"))
ets_viz_path = os.path.join(week_path(visualizations_path, 2, "ets"))
prophet_viz_path = os.path.join(week_path(visualizations_path, 2, "prophet"))
comparison_path = os.path.join(week_path(visualizations_path, 2, "comparison"))

# results Paths
arima_results_path = os.path.join(week_path(results_path, 2, "arima"))
sarima_results_path = os.path.join(week_path(results_path, 2, "sarima"))
ets_results_path = os.path.join(week_path(results_path, 2, "ets"))
prophet_results_path = os.path.join(week_path(results_path, 2, "prophet"))
comparison_results_path = os.path.join(week_path(results_path, 2, 'comparison'))

# ML results Paths
linear_results_path = os.path.join(week_path(results_path, 3, "linear"))
svr_results_path = os.path.join(week_path(results_path, 3, "svr"))
random_forest_results_path = os.path.join(week_path(results_path, 3, "random_forest"))
xgboost_results_path = os.path.join(week_path(results_path, 3, "xgboost"))

#ML Visualistaion Paths
linear_viz_paths = arima_viz_path = os.path.join(week_path(visualizations_path, 3, "linear"))
svr_viz_paths = arima_viz_path = os.path.join(week_path(visualizations_path, 3, "svr"))
random_forest_viz_paths = arima_viz_path = os.path.join(week_path(visualizations_path, 3, "random_forest"))
xgboost_viz_paths = arima_viz_path = os.path.join(week_path(visualizations_path, 3, "xgboost"))

# ML Models Paths
linear_models_path = os.path.join(week_path(models_path, 3, "linear"))
svr_models_path = os.path.join(week_path(models_path, 3, "svr"))
random_forest_models_path = os.path.join(week_path(models_path, 3, "random_forest"))
xgboost_models_path = os.path.join(week_path(models_path, 3, "xgboost"))

# Deep Learning Paths
lstm_model_path = os.path.join(week_path(models_path, 3, "lstm"))
lstm_result_path = os.path.join(week_path(results_path, 3, "lstm"))
lstm_viz_path = os.path.join(week_path(visualizations_path, 3, "lstm"))
# =======================================
# Helper Functions
# =======================================

def get_path(path_type: str, week: int = None) -> str:
    """
    Convenient path resolver with automatic directory creation.

    Args:
        path_type (str): Keyword for the path.
        week (int, optional): Week number for week-specific paths.

    Returns:
        str: Resolved absolute path.
    """
    paths: Dict[str, str] = {
        # Project root structure
        "project": project_root,

        # Config
        "config": config_path,

        # Data Folder
        "data": data_path,
        "csv": csv_path,
        "raw": raw_path,
        "processed": processed_path,
        "time_series": time_series_path,
        "filtered": filtered_path,

        # Reports Folder
        "reports": reports_path,
        "visualizations": visualizations_path,
        "results": results_path,
        "mlflow": mlflow_path,
        
        # Models Folder
        "models": models_path,
        "arima_models": arima_models_path,
        "sarima_models": sarima_models_path,
        "ets_models": ets_models_path,
        "prophet_models": prophet_models_path,
        
        # Visualizations for Models
        "arima_viz": arima_viz_path,
        "sarima_viz": sarima_viz_path,
        "ets_viz": ets_viz_path,
        "prophet_viz": prophet_viz_path,
        "comparison_viz": comparison_path,
        
        "linear_viz": linear_viz_paths,
        "svr_viz": svr_viz_paths,
        "random_forest_viz": random_forest_viz_paths,
        "xgboost_viz": xgboost_viz_paths,
        
        # Results for Models
        "arima_results": arima_results_path,
        "sarima_results": sarima_results_path,
        "ets_results": ets_results_path,
        "prophet_results": prophet_results_path,
        "comparison_results": comparison_results_path,
        
        # ML Models Paths
        "linear_models": linear_models_path,
        "svr_models": svr_models_path,
        "random_forest_models": random_forest_models_path,
        "xgboost_models": xgboost_models_path,
        
        # ML Results Paths
        "linear_results": linear_results_path,
        "svr_results": svr_results_path,
        "random_forest_results": random_forest_results_path,
        "xgboost_results": xgboost_results_path,
        
        # Deep Learning
        "lstm_model":lstm_model_path,
        "lstm_results": lstm_result_path,
        "lstm_viz":lstm_viz_path
    }

    # Handle week-specific paths dynamically
    if path_type in {"features", "loader_processed", "eda", "eda_stats", "loader", 'cleaned', "features_results", "features_viz", }:
        if week is None:
            raise ValueError(f"❌ 'week' argument required for path type '{path_type}'")
        if path_type == "features":
            resolved = week_path(processed_path, week, "features")
        elif path_type == "loader_processed":
            resolved = week_path(processed_path, week, "loader_processed")
        elif path_type == "cleaned":
            resolved = week_path(processed_path, week, "cleaned")
        elif path_type == "eda":
            resolved = week_path(visualizations_path, week, "eda")
        elif path_type == "features_results":
            resolved = week_path(results_path, week, "features_results")
        elif path_type == "features_viz":
            resolved = week_path(visualizations_path, week, "features_viz")
        elif path_type == "eda_stats":
            resolved = week_path(results_path, week, "eda_stats")
        elif path_type == "loader":
            resolved = week_path(visualizations_path, week, "loader")
    else:
        if path_type not in paths:
            raise ValueError(
                f"❌ Unknown path type '{path_type}'. Allowed values: {list(paths.keys()) + ['features','eda','eda_stats','loader']}"
            )
        resolved = paths[path_type]

    resolved = os.path.abspath(resolved)
    os.makedirs(resolved, exist_ok=True)
    return resolved


def load_yaml(file_path: str) -> dict:
    """
    General-purpose YAML loader.

    Args:
        file_path (str): Path to the YAML file.

    Returns:
        dict: Parsed YAML content as a Python dictionary.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"❌ YAML file not found: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        try:
            config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise RuntimeError(f"❌ Failed to parse YAML file {file_path}: {e}") from e

    if not isinstance(config, dict):
        raise ValueError(f"❌ YAML file {file_path} must contain a dictionary at the root.")

    return config

# =======================================
# NEW: Model Persistence Functions
# =======================================

def save_model(model: Any, model_type: str, week: int, filename: str) -> str:
    """
    Saves a trained model object (e.g., ARIMA, Prophet) using pickle.

    Args:
        model (Any): The trained model object to save.
        model_type (str): The type of model (e.g., 'arima', 'sarima', 'prophet').
        week (int): The current week number (determines path).
        filename (str): The name for the saved file (e.g., 'best_arima_p3q1.pkl').

    Returns:
        str: The absolute path where the model was saved.
    """
    # 1. Determine the correct directory path based on model type and week
    path_key = f"{model_type}_models"
    try:
        save_dir = get_path(path_key, week=week)
    except ValueError:
        print(f"[WARN] Unknown model type '{model_type}'. Saving to general models path.")
        save_dir = get_path("models", week=week)
        
    # 2. Construct the full save path
    save_path = os.path.join(save_dir, filename)

    # 3. Save the model using pickle
    print(f"💾 Attempting to save model to: {save_path}")
    try:
        with open(save_path, 'wb') as f:
            pickle.dump(model, f)
        print("✅ Model successfully saved.")
        return save_path
    except Exception as e:
        print(f"❌ Failed to save model to {save_path}. Error: {e}")
        return ""
        
def load_model(model_type: str, week: int, filename: str) -> Any:
    """
    Loads a saved model object from disk using pickle.

    Args:
        model_type (str): The type of model (e.g., 'arima', 'sarima').
        week (int): The current week number.
        filename (str): The name of the saved file (e.g., 'best_arima_p3q1.pkl').

    Returns:
        Any: The loaded model object, or None if loading fails.
    """
    path_key = f"{model_type}_models"
    try:
        load_dir = get_path(path_key, week=week)
    except ValueError:
        load_dir = get_path("models", week=week)

    load_path = os.path.join(load_dir, filename)

    if not os.path.exists(load_path):
        print(f"❌ Model file not found at: {load_path}")
        return None

    print(f"🔄 Attempting to load model from: {load_path}")
    try:
        with open(load_path, 'rb') as f:
            model = pickle.load(f)
        print("✅ Model successfully loaded.")
        return model
    except Exception as e:
        print(f"❌ Failed to load model from {load_path}. Error: {e}")
        return None