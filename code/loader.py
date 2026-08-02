from pathlib import Path
from typing import Dict, Optional, Union

import pandas as pd


class DataLoader:
    """Load and cache the challenge datasets once for reuse across the pipeline."""

    REQUIRED_DATASETS = (
        "messages.csv",
        "users.csv",
        "groups.csv",
        "group_members.csv",
        "business_accounts.csv",
        "user_business_history.csv",
        "message_history.csv",
        "message_events.csv",
        "images.csv",
        "voice_notes.csv",
        "daily_notification_summary.csv",
    )

    def __init__(self, dataset_dir: Optional[Union[str, Path]] = None) -> None:
        """Initialize the loader and eagerly populate the cached dataset attributes."""
        self._dataset_dir = self._resolve_dataset_dir(dataset_dir)
        self._validate_dataset_dir()
        self._cache: Dict[str, pd.DataFrame] = {}

        self.messages = self.get_messages()
        self.users = self.get_users()
        self.groups = self.get_groups()
        self.group_members = self.get_group_members()
        self.business_accounts = self.get_business_accounts()
        self.user_business_history = self.get_user_business_history()
        self.message_history = self.get_message_history()
        self.message_events = self.get_message_events()
        self.images = self.get_images()
        self.voice_notes = self.get_voice_notes()
        self.daily_summary = self.get_daily_summary()

    def _resolve_dataset_dir(self, dataset_dir: Optional[Union[str, Path]]) -> Path:
        """Resolve the dataset directory relative to the repository root."""
        if dataset_dir is None:
            return Path(__file__).resolve().parents[1] / "dataset"

        path = Path(dataset_dir)
        if not path.is_absolute():
            path = Path(__file__).resolve().parents[1] / path

        return path.resolve()

    def _validate_dataset_dir(self) -> None:
        """Ensure the dataset directory exists and contains all required files."""
        if not self._dataset_dir.exists():
            raise FileNotFoundError(f"Dataset directory does not exist: {self._dataset_dir}")

        if not self._dataset_dir.is_dir():
            raise NotADirectoryError(f"Dataset path is not a directory: {self._dataset_dir}")

        missing_files = [
            filename for filename in self.REQUIRED_DATASETS if not (self._dataset_dir / filename).is_file()
        ]
        if missing_files:
            missing_list = ", ".join(missing_files)
            raise FileNotFoundError(f"Required dataset files are missing: {missing_list}")

    def _load_dataframe(self, filename: str) -> pd.DataFrame:
        """Load a CSV file and cache it so each dataset is read once."""
        if filename in self._cache:
            return self._cache[filename]

        file_path = self._dataset_dir / filename
        try:
            dataframe = pd.read_csv(file_path)
        except Exception as exc:  # pragma: no cover - defensive error wrapping
            raise ValueError(f"Failed to load dataset '{filename}' from '{file_path}': {exc}") from exc

        self._cache[filename] = dataframe
        return dataframe

    def get_dataframe(self, filename: str) -> pd.DataFrame:
        """Return a DataFrame for a dataset file by name."""
        if not filename.endswith(".csv"):
            raise ValueError(f"Expected a CSV filename, received: {filename}")

        return self._load_dataframe(filename)

    def get_messages(self) -> pd.DataFrame:
        """Return the messages dataset."""
        return self.get_dataframe("messages.csv")

    def get_users(self) -> pd.DataFrame:
        """Return the users dataset."""
        return self.get_dataframe("users.csv")

    def get_groups(self) -> pd.DataFrame:
        """Return the groups dataset."""
        return self.get_dataframe("groups.csv")

    def get_group_members(self) -> pd.DataFrame:
        """Return the group-members mapping dataset."""
        return self.get_dataframe("group_members.csv")

    def get_business_accounts(self) -> pd.DataFrame:
        """Return the business account dataset."""
        return self.get_dataframe("business_accounts.csv")

    def get_user_business_history(self) -> pd.DataFrame:
        """Return the user-business interaction history dataset."""
        return self.get_dataframe("user_business_history.csv")

    def get_message_history(self) -> pd.DataFrame:
        """Return the message history dataset."""
        return self.get_dataframe("message_history.csv")

    def get_message_events(self) -> pd.DataFrame:
        """Return the message events dataset."""
        return self.get_dataframe("message_events.csv")

    def get_images(self) -> pd.DataFrame:
        """Return the image metadata dataset."""
        return self.get_dataframe("images.csv")

    def get_voice_notes(self) -> pd.DataFrame:
        """Return the voice-note metadata dataset."""
        return self.get_dataframe("voice_notes.csv")

    def get_daily_summary(self) -> pd.DataFrame:
        """Return the daily notification summary dataset."""
        return self.get_dataframe("daily_notification_summary.csv")

    def clear_cache(self) -> None:
        """Clear the cached DataFrames, forcing them to be reloaded on next access."""
        self._cache.clear()