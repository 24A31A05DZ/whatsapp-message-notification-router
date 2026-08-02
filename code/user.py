from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import pandas as pd


@dataclass(frozen=True)
class UserProfile:
    """Structured user preferences and notification behavior."""

    user_id: Optional[str]
    quiet_hours: str
    notification_behavior: str
    open_rate: float
    dismiss_rate: float
    reply_rate: float
    current_notification_load: int
    preferences: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        """Return a plain dictionary representation for downstream logic."""
        return {
            "user_id": self.user_id,
            "quiet_hours": self.quiet_hours,
            "notification_behavior": self.notification_behavior,
            "open_rate": self.open_rate,
            "dismiss_rate": self.dismiss_rate,
            "reply_rate": self.reply_rate,
            "current_notification_load": self.current_notification_load,
            "preferences": self.preferences,
        }


class UserService:
    """Provide user-based notification preferences from the dataset."""

    def __init__(self, data_dir: Optional[Union[str, Path]] = None) -> None:
        self._data_dir = self._resolve_data_dir(data_dir)
        self.users = self._load_users()

    def _resolve_data_dir(self, data_dir: Optional[Union[str, Path]]) -> Path:
        """Resolve the dataset directory relative to the repository root."""
        if data_dir is None:
            base_dir = Path(__file__).resolve().parents[1]
            return base_dir / "dataset"

        path = Path(data_dir)
        if not path.is_absolute():
            base_dir = Path(__file__).resolve().parents[1]
            path = base_dir / path

        return path.resolve()

    def _load_users(self) -> pd.DataFrame:
        """Load the users dataset from disk."""
        file_path = self._data_dir / "users.csv"
        if not file_path.exists():
            raise FileNotFoundError(f"Users dataset not found: {file_path}")

        users = pd.read_csv(file_path)
        for column in ["messages_opened_30d", "messages_replied_30d", "notifications_dismissed_30d", "messages_reported_30d"]:
            users[column] = pd.to_numeric(users[column], errors="coerce").fillna(0)

        return users

    def _get_user_row(self, user_id: Optional[Union[str, int]]) -> Optional[pd.Series]:
        """Return the user row matching the provided id, if any."""
        if user_id is None or pd.isna(user_id):
            return None

        normalized_id = str(user_id)
        matches = self.users[self.users["user_id"] == normalized_id]
        if matches.empty:
            return None

        return matches.iloc[0]

    def _derive_notification_behavior(self, dismiss_rate: float, reply_rate: float, open_rate: float) -> str:
        """Classify notification behavior based on recent engagement signals."""
        if dismiss_rate >= 0.6:
            return "dismiss-heavy"
        if reply_rate >= 0.4:
            return "engaged"
        if open_rate >= 0.8:
            return "curious"
        return "passive"

    def get_profile(self, user_id: Optional[Union[str, int]]) -> UserProfile:
        """Return a structured user profile with preference and behavior signals."""
        user = self._get_user_row(user_id)

        if user is None:
            return UserProfile(
                user_id=str(user_id) if user_id is not None and not pd.isna(user_id) else None,
                quiet_hours="unknown",
                notification_behavior="unknown",
                open_rate=0.0,
                dismiss_rate=0.0,
                reply_rate=0.0,
                current_notification_load=0,
                preferences={},
            )

        notifications_dismissed = int(user.get("notifications_dismissed_30d", 0))
        messages_opened = int(user.get("messages_opened_30d", 0))
        messages_replied = int(user.get("messages_replied_30d", 0))
        messages_reported = int(user.get("messages_reported_30d", 0))

        total_messages = max(1, messages_opened + messages_replied + notifications_dismissed + messages_reported)
        open_rate = round(messages_opened / total_messages, 3)
        dismiss_rate = round(notifications_dismissed / total_messages, 3)
        reply_rate = round(messages_replied / total_messages, 3)

        quiet_hours = str(user.get("do_not_disturb_window", "unknown"))
        notification_behavior = self._derive_notification_behavior(dismiss_rate, reply_rate, open_rate)

        preferences = {
            "quiet_hours": quiet_hours,
            "avoid_silent_dismiss": dismiss_rate > 0.4,
            "prefers_short_replies": reply_rate < 0.2,
            "reports_suspicious_content": messages_reported > 0,
        }

        return UserProfile(
            user_id=str(user["user_id"]),
            quiet_hours=quiet_hours,
            notification_behavior=notification_behavior,
            open_rate=open_rate,
            dismiss_rate=dismiss_rate,
            reply_rate=reply_rate,
            current_notification_load=notifications_dismissed + messages_replied + messages_opened,
            preferences=preferences,
        )

    def get_quiet_hours(self, user_id: Optional[Union[str, int]]) -> str:
        """Return the user's quiet hours window."""
        return self.get_profile(user_id).quiet_hours

    def get_notification_behavior(self, user_id: Optional[Union[str, int]]) -> str:
        """Return the user's notification behavior label."""
        return self.get_profile(user_id).notification_behavior

    def get_open_rate(self, user_id: Optional[Union[str, int]]) -> float:
        """Return the user's open rate."""
        return self.get_profile(user_id).open_rate

    def get_dismiss_rate(self, user_id: Optional[Union[str, int]]) -> float:
        """Return the user's dismiss rate."""
        return self.get_profile(user_id).dismiss_rate

    def get_reply_rate(self, user_id: Optional[Union[str, int]]) -> float:
        """Return the user's reply rate."""
        return self.get_profile(user_id).reply_rate

    def get_current_notification_load(self, user_id: Optional[Union[str, int]]) -> int:
        """Return the current notification load estimate."""
        return self.get_profile(user_id).current_notification_load

    def get_preferences(self, user_id: Optional[Union[str, int]]) -> dict[str, object]:
        """Return the user's inferred preferences."""
        return self.get_profile(user_id).preferences
