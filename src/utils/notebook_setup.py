# src/utils/notebook_setup.py
"""
Setup notebook environment for retail demand analysis project.
"""


import os
import sys
from IPython import get_ipython # type: ignore

def setup_notebook(project_subfolder_level=2, bad_paths=None):
    """
    Setup notebook environment:
      • Cleans sys.path of unwanted paths
      • Adds project root to sys.path
      • Enables autoreload
      • Initializes Plotly offline mode
      • Verifies that `src` module is importable

    Args:
        project_subfolder_level (int): How many folders up to find project root.
            E.g., if notebook is in notebooks/week_1/, set level=2.
        bad_paths (list[str]): List of paths to remove from sys.path.

    Returns:
        str: The absolute path of the project root
    """
    # -----------------------------
    # Remove bad paths from sys.path
    # -----------------------------
    if bad_paths is None:
        bad_paths = [
            "/Users/guykaptue/my_work_spaces/machine learning",
            "/Users/guykaptue/my_work_spaces/machine learning/masterschool/time-series/retail_demand_analysis/src"
        ]

    sys.path = [p for p in sys.path if os.path.abspath(p) not in bad_paths]

    # -----------------------------
    # Enable autoreload if in IPython
    # -----------------------------
    ipython = get_ipython()
    if ipython is not None:
        ipython.run_line_magic("load_ext", "autoreload")
        ipython.run_line_magic("autoreload", "2")
        ipython.run_line_magic("matplotlib", "inline")

    # -----------------------------
    # Determine project root dynamically
    # -----------------------------
    cwd = os.getcwd()
    project_root = os.path.abspath(os.path.join(cwd, *[".."]*project_subfolder_level))

    # Add project root to sys.path
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    # -----------------------------
    # Initialize Plotly offline mode
    # -----------------------------
    import plotly.offline as pyo # type: ignore
    pyo.init_notebook_mode(connected=True)

    # -----------------------------
    # Verify that src is importable
    # -----------------------------
    try:
        import src
        print("✅ src module found at:", src.__file__)
    except ModuleNotFoundError:
        print("❌ src module not found. Check your sys.path!")

    # -----------------------------
    # Print info
    # -----------------------------
    print("✅ Notebook environment setup complete")
    print("Project root:", project_root)
    print("sys.path (first 5 entries):", sys.path[:5])
    print("src folder exists:", os.path.exists(os.path.join(project_root, "src")))

    return project_root
