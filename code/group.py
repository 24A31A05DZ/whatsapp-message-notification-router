from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import pandas as pd


@dataclass(frozen=True)
class GroupContext:
    """Structured context for a user's group relationship."""

    group_id: Optional[str]
    group_name: str
    group_type: str
    muted_group: bool
    admin_messages: bool
    high_priority_group: bool
    school_group: bool
    family_group: bool
    society_group: bool
    activity_score: float
    user_engagement: float
    member_count: int
    admin_count: int
    messages_30d: int

    def to_dict(self) -> dict[str, object]:
        """Return a plain dictionary for downstream routing logic."""
        return {
            "group_id": self.group_id,
            "group_name": self.group_name,
            "group_type": self.group_type,
            "muted_group": self.muted_group,
            "admin_messages": self.admin_messages,
            "high_priority_group": self.high_priority_group,
            "school_group": self.school_group,
            "family_group": self.family_group,
            "society_group": self.society_group,
            "activity_score": self.activity_score,
            "user_engagement": self.user_engagement,
            "member_count": self.member_count,
            "admin_count": self.admin_count,
            "messages_30d": self.messages_30d,
        }


class GroupService:
    """Provide group context using group metadata and per-user membership data."""

    def __init__(self, data_dir: Optional[Union[str, Path]] = None) -> None:
        self._data_dir = self._resolve_data_dir(data_dir)
        self.groups = self._load_groups()
        self.members = self._load_members()

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

    def _load_groups(self) -> pd.DataFrame:
        """Load group metadata from disk."""
        file_path = self._data_dir / "groups.csv"
        if not file_path.exists():
            raise FileNotFoundError(f"Groups dataset not found: {file_path}")

        groups = pd.read_csv(file_path)
        for column in ["member_count", "admin_count", "messages_30d"]:
            groups[column] = pd.to_numeric(groups[column], errors="coerce").fillna(0)

        return groups

    def _load_members(self) -> pd.DataFrame:
        """Load the membership matrix from disk."""
        file_path = self._data_dir / "group_members.csv"
        if not file_path.exists():
            raise FileNotFoundError(f"Group members dataset not found: {file_path}")

        members = pd.read_csv(file_path)
        for column in ["messages_sent_30d", "messages_read_30d", "replies_sent_30d", "notifications_dismissed_30d"]:
            members[column] = pd.to_numeric(members[column], errors="coerce").fillna(0)

        members["group_muted_by_user"] = pd.to_numeric(members["group_muted_by_user"], errors="coerce").fillna(0).astype(int)
        return members

    def _get_group_row(self, group_id: Optional[Union[str, int]]) -> Optional[pd.Series]:
        """Return the group metadata entry for the provided id, if present."""
        if group_id is None or pd.isna(group_id):
            return None

        matches = self.groups[self.groups["group_id"] == str(group_id)]
        if matches.empty:
            return None

        return matches.iloc[0]

    def _get_user_membership(self, user_id: Optional[Union[str, int]], group_id: Optional[Union[str, int]]) -> Optional[pd.Series]:
        """Return the membership row for a user in a group, if present."""
        if user_id is None or pd.isna(user_id) or group_id is None or pd.isna(group_id):
            return None

        matches = self.members[
            (self.members["user_id"] == str(user_id)) & (self.members["group_id"] == str(group_id))
        ]
        if matches.empty:
            return None

        return matches.iloc[0]

    def _classify_group_type(self, group_type: str) -> tuple[bool, bool, bool, bool]:
        """Infer whether the group is family, school, society, or otherwise."""
        normalized = str(group_type or "").lower()
        is_school_group = "school" in normalized or normalized == "school_group"
        is_family_group = normalized in {"family", "extended_family"}
        is_society_group = normalized in {"society", "residents"}
        return is_school_group, is_family_group, is_society_group, False

    def get_context(self, user_id: Optional[Union[str, int]], group_id: Optional[Union[str, int]]) -> GroupContext:
        """Return a structured group context object for a user-group pair."""
        group = self._get_group_row(group_id)
        membership = self._get_user_membership(user_id, group_id)

        if group is None:
            return GroupContext(
                group_id=str(group_id) if group_id is not None and not pd.isna(group_id) else None,
                group_name="Unknown",
                group_type="unknown",
                muted_group=False,
                admin_messages=False,
                high_priority_group=False,
                school_group=False,
                family_group=False,
                society_group=False,
                activity_score=0.0,
                user_engagement=0.0,
                member_count=0,
                admin_count=0,
                messages_30d=0,
            )

        group_type = str(group.get("group_type", "unknown"))
        school_group, family_group, society_group, _ = self._classify_group_type(group_type)

        muted_group = bool(membership is not None and int(membership.get("group_muted_by_user", 0)) == 1)
        admin_messages = bool(membership is not None and str(membership.get("role", "")).lower() == "admin")
        member_count = int(group.get("member_count", 0))
        admin_count = int(group.get("admin_count", 0))
        messages_30d = int(group.get("messages_30d", 0))

        activity_score = round(min(1.0, messages_30d / max(1000, member_count * 10)), 3)
        user_engagement = 0.0
        if membership is not None:
            read_ratio = float(membership.get("messages_read_30d", 0)) / max(1, int(membership.get("messages_sent_30d", 0)) + int(membership.get("messages_read_30d", 0)))
            reply_ratio = float(membership.get("replies_sent_30d", 0)) / max(1, int(membership.get("messages_sent_30d", 0)))
            dismissal_ratio = float(membership.get("notifications_dismissed_30d", 0)) / max(1, int(membership.get("messages_sent_30d", 0)) + int(membership.get("messages_read_30d", 0)))
            user_engagement = round(min(1.0, 0.55 * read_ratio + 0.3 * reply_ratio + 0.15 * (1 - dismissal_ratio)), 3)

        high_priority_group = muted_group or admin_messages or member_count >= 100 or messages_30d >= 500

        return GroupContext(
            group_id=str(group["group_id"]),
            group_name=str(group.get("group_name", "Unknown")),
            group_type=group_type,
            muted_group=muted_group,
            admin_messages=admin_messages,
            high_priority_group=high_priority_group,
            school_group=school_group,
            family_group=family_group,
            society_group=society_group,
            activity_score=activity_score,
            user_engagement=user_engagement,
            member_count=member_count,
            admin_count=admin_count,
            messages_30d=messages_30d,
        )

    def is_muted_group(self, user_id: Optional[Union[str, int]], group_id: Optional[Union[str, int]]) -> bool:
        """Return True if the user has muted the group."""
        return self.get_context(user_id, group_id).muted_group

    def has_admin_messages(self, user_id: Optional[Union[str, int]], group_id: Optional[Union[str, int]]) -> bool:
        """Return True when the membership indicates admin-level involvement."""
        return self.get_context(user_id, group_id).admin_messages

    def is_high_priority_group(self, user_id: Optional[Union[str, int]], group_id: Optional[Union[str, int]]) -> bool:
        """Return True when the group should be treated as high priority."""
        return self.get_context(user_id, group_id).high_priority_group

    def is_school_group(self, group_id: Optional[Union[str, int]]) -> bool:
        """Return True when the group looks like a school group."""
        return self.get_context(None, group_id).school_group

    def is_family_group(self, group_id: Optional[Union[str, int]]) -> bool:
        """Return True when the group resembles a family group."""
        return self.get_context(None, group_id).family_group

    def is_society_group(self, group_id: Optional[Union[str, int]]) -> bool:
        """Return True when the group resembles a society group."""
        return self.get_context(None, group_id).society_group

    def get_activity_score(self, group_id: Optional[Union[str, int]]) -> float:
        """Return the normalized activity score for the group."""
        return self.get_context(None, group_id).activity_score

    def get_user_engagement(self, user_id: Optional[Union[str, int]], group_id: Optional[Union[str, int]]) -> float:
        """Return the user's engagement score for the group."""
        return self.get_context(user_id, group_id).user_engagement
