# app/bootstrap.py
import os  # noqa: F401
import sys
from pathlib import Path

# Determine project root (parent of /app)
APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent

# Add project root to sys.path
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Optional: verify src exists
SRC_DIR = PROJECT_ROOT / "src"
if not SRC_DIR.exists():
    print("⚠️ Warning: src directory not found at:", SRC_DIR)
