import os
import io
import pandas as pd
import requests # type: ignore
from src.utils import get_path, load_yaml


class GoogleDriveLoader:
    """Handles downloading and loading CSVs directly from Google Drive."""

    @staticmethod
    def _make_drive_url(file_id: str) -> str:
        """Build the direct download URL from a Google Drive file ID."""
        return f"https://drive.google.com/uc?id={file_id}"

    @staticmethod
    def _load_csv_from_url(url: str) -> pd.DataFrame:
        """Load a CSV directly from a Google Drive URL into a DataFrame."""
        response = requests.get(url)
        response.raise_for_status()
        return pd.read_csv(io.StringIO(response.text))

    def download_csv(self, file_id: str, output_name: str, folder_name: str = "raw") -> str:
        """Download a CSV from Google Drive and save it locally."""
        url = self._make_drive_url(file_id)
        path = get_path(folder_name)
        os.makedirs(path, exist_ok=True)
        output_path = os.path.join(path, output_name)

        print(f"📥 Downloading {output_name} from Google Drive file ID {file_id}...")
        response = requests.get(url)
        response.raise_for_status()

        with open(output_path, "wb") as f:
            f.write(response.content)

        print(f"✅ Saved {output_name} to {output_path}")
        return output_path

    def download_csvs(self, file_ids: dict[str, str], folder_name: str = "raw", skip: list[str] = None) -> dict[str, str]:
        """Download multiple CSVs from Google Drive and save them locally."""
        skip = skip or []
        saved_files = {}
        for name, fid in file_ids.items():
            if name in skip:
                print(f"⏭️ Skipping {name}")
                continue
            output_name = f"{name}.csv"
            saved_files[name] = self.download_csv(fid, output_name, folder_name)
        return saved_files

    def download_config(self, yaml_filename: str = "gdrive_file_ids.yaml", folder_name: str = "raw", skip: list[str] = None) -> dict[str, str]:
        """Download multiple CSVs from Google Drive using a YAML config file."""
        yaml_path = os.path.join(get_path("config"), yaml_filename)
        config = load_yaml(yaml_path)
        if "file_ids" not in config:
            raise KeyError(f"❌ YAML file {yaml_path} must contain a 'file_ids' dictionary at the root.")
        return self.download_csvs(config["file_ids"], folder_name=folder_name, skip=skip)

    def load_metadata_files(self, file_ids: dict[str, str]) -> dict[str, pd.DataFrame]:
        """
        Load metadata CSVs (excluding train.csv) directly into DataFrames.

        Args:
            file_ids (dict): Mapping of {name: file_id}.

        Returns:
            dict[str, pd.DataFrame]: Mapping of {name: DataFrame}.
        """
        metadata_keys = ["holiday_events", "items", "oil", "stores", "transactions"]
        dfs = {}
        for key in metadata_keys:
            if key not in file_ids:
                raise KeyError(f"❌ Missing '{key}' in file_ids config.")
            url = self._make_drive_url(file_ids[key])
            dfs[key] = self._load_csv_from_url(url)
            print(f"✅ Loaded {key}.csv into DataFrame (shape={dfs[key].shape})")
        return dfs

    def load_metadata_from_config(self, yaml_filename: str = "gdrive_file_ids.yaml") -> dict[str, pd.DataFrame]:
        """
        Load metadata CSVs directly into DataFrames using a YAML config file.

        Args:
            yaml_filename (str): Name of YAML config file inside the config folder.

        Returns:
            dict[str, pd.DataFrame]: Mapping of {name: DataFrame}.
        """
        yaml_path = os.path.join(get_path("config"), yaml_filename)
        config = load_yaml(yaml_path)
        if "file_ids" not in config:
            raise KeyError(f"❌ YAML file {yaml_path} must contain a 'file_ids' dictionary at the root.")
        return self.load_metadata_files(config["file_ids"])
