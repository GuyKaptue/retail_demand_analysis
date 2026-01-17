# src/week_1/processor/kaggle_loader.py
import os
import requests  # type: ignore
import zipfile
from kaggle.api.kaggle_api_extended import KaggleApi  # type: ignore
from src.utils import get_path

class KaggleDataLoader:
    """
    Loader for Kaggle datasets and competitions.
    Handles authentication, download, unzip, and file listing.
    """

    def __init__(self):
        """Initialize Kaggle API."""
        self.api = KaggleApi()
        try:
            self.api.authenticate()
        except Exception as e:
            raise RuntimeError(
                "❌ Kaggle API authentication failed. Ensure ~/.kaggle/kaggle.json "
                "or KAGGLE_USERNAME/KAGGLE_KEY are set up correctly."
            ) from e

    # -------------------------------
    # Kaggle download/unzip utilities
    # -------------------------------
    def _detect_mode(self, slug: str) -> str:
        """Detect whether slug refers to a competition or dataset."""
        slug = slug.strip("/")
        if slug.startswith("competitions/"):
            return "competition"
        elif "/" in slug:
            return "dataset"
        return "competition"

    def _zip_name(self, slug: str, mode: str) -> str:
        """Return the expected zip filename for a slug."""
        slug = slug.strip("/")
        if mode == "dataset" and "/" in slug:
            return slug.split("/")[-1] + ".zip"
        else:
            return slug.replace("competitions/", "") + ".zip"

    def _has_csvs(self, path: str) -> bool:
        """Check if folder contains any CSV files."""
        return any(f.endswith(".csv") for f in os.listdir(path))

    def _check_competition_joined(self, comp_name: str) -> None:
        """
        Check if the user has joined the competition.
        Raises RuntimeError if not joined.
        """
        comps = self.api.competitions_list(search=comp_name)
        for comp in comps:
            if comp.ref == comp_name and not comp.userHasEntered:
                raise RuntimeError(
                    f"❌ You have not joined the competition '{comp_name}'.\n"
                    f"👉 Visit https://www.kaggle.com/c/{comp_name} and click 'Join Competition'."
                )

    def download(self, slug: str, folder_name: str = "raw", force: bool = False, mode: str = None) -> str:
        """Download dataset or competition files from Kaggle."""
        path = get_path(folder_name)
        os.makedirs(path, exist_ok=True)

        slug = slug.strip("/")
        mode = mode or self._detect_mode(slug)
        zip_name = self._zip_name(slug, mode)
        zip_path = os.path.join(path, zip_name)

        # Skip download if CSVs already exist
        if self._has_csvs(path) and not force:
            print(f"⚡ Found existing CSV files in {path}, skipping download.")
            return zip_path

        print(f"🔎 Mode detected: {mode}")
        print(f"📥 Downloading {slug} ({mode}) to {path}...")

        try:
            if mode == "competition":
                comp_name = slug.replace("competitions/", "")
                # Pre-check: ensure user has joined
                self._check_competition_joined(comp_name)
                self.api.competition_download_files(comp_name, path=path)
            elif mode == "dataset":
                self.api.dataset_download_files(slug, path=path)
            else:
                raise ValueError("mode must be 'competition' or 'dataset'")
            print("✅ Download complete.")
        except requests.exceptions.HTTPError as e:
            if e.response is not None and e.response.status_code in (401, 403):
                raise RuntimeError(
                    f"❌ Unauthorized/Forbidden to download '{slug}' ({mode}).\n"
                    f"👉 If competition: join on Kaggle.\n"
                    f"👉 If dataset: ensure it's public and accessible."
                ) from e
            else:
                raise
        return zip_path

    def unzip(self, slug: str, folder_name: str = "raw", mode: str = None) -> str:
        """Unzip the downloaded dataset into the given folder and delete the zip file afterwards."""
        path = get_path(folder_name)
        slug = slug.strip("/")
        mode = mode or self._detect_mode(slug)
        zip_name = self._zip_name(slug, mode)
        zip_path = os.path.join(path, zip_name)

        if not os.path.exists(zip_path):
            raise FileNotFoundError(f"❌ {zip_path} not found. Run download() first.")

        print(f"📦 Extracting {zip_path} to {path}...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(path)
        print("✅ Extraction complete.")

        # Delete the zip file after extraction
        try:
            os.remove(zip_path)
            print(f"🗑️ Deleted archive {zip_path}")
        except Exception as e:
            print(f"⚠️ Could not delete {zip_path}: {e}")

        # Verify extracted files
        extracted = [f for f in os.listdir(path) if f.endswith(".csv")]
        print(f"📂 Extracted {len(extracted)} CSV files: {extracted}")

        return path

    def prepare(self, slug: str, folder_name: str = "raw", force: bool = False, mode: str = None) -> str:
        """Convenience method: download + unzip in one call."""
        slug = slug.strip("/")
        mode = mode or self._detect_mode(slug)

        # Skip if CSVs already exist
        path = get_path(folder_name)
        if os.path.exists(path) and self._has_csvs(path) and not force:
            print(f"⚡ Found existing CSV files in {path}, skipping download/unzip.")
            return path

        self.download(slug, folder_name=folder_name, force=force, mode=mode)
        return self.unzip(slug, folder_name=folder_name, mode=mode)

    def list_files(self, folder_name: str = "raw") -> list[str]:
        """List all files in the given folder (excluding zip files)."""
        path = get_path(folder_name)
        files = [
            f for f in os.listdir(path)
            if os.path.isfile(os.path.join(path, f)) and not f.endswith(".zip")
        ]
        return sorted(files)
