from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd


class HistoryService:
    """Retrieve historical evidence and user engagement signals without machine learning."""

    def __init__(self, data_dir: Optional[Union[str, Path]] = None) -> None:
        self._data_dir = self._resolve_data_dir(data_dir)
        self.history = self._load_history()
        self.events = self._load_events()

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

    def _load_history(self) -> pd.DataFrame:
        """Load message history and keep the DataFrame ready for reuse."""
        file_path = self._data_dir / "message_history.csv"
        if not file_path.exists():
            raise FileNotFoundError(f"Message history dataset not found: {file_path}")
        return pd.read_csv(file_path)

    def _load_events(self) -> pd.DataFrame:
        """Load message event logs and normalize them for analytics."""
        file_path = self._data_dir / "message_events.csv"
        if not file_path.exists():
            raise FileNotFoundError(f"Message events dataset not found: {file_path}")

        events = pd.read_csv(file_path)
        for column in ["message_opened", "message_replied", "notification_dismissed", "muted_after_message", "message_reported"]:
            events[column] = pd.to_numeric(events[column], errors="coerce").fillna(0)

        return events

    @staticmethod
    def _normalize_text(text: Optional[Union[str, Any]]) -> str:
        """Normalize text into lowercase tokens for simple rule-based matching."""
        if text is None or pd.isna(text):
            return ""
        return re.sub(r"[^a-z0-9]+", " ", str(text).lower()).strip()

    @staticmethod
    def _token_overlap_score(text_a: Optional[Union[str, Any]], text_b: Optional[Union[str, Any]]) -> float:
        """Compute a simple token overlap score between two strings."""
        tokens_a = set(HistoryService._normalize_text(text_a).split())
        tokens_b = set(HistoryService._normalize_text(text_b).split())

        if not tokens_a or not tokens_b:
            return 0.0

        return round(len(tokens_a & tokens_b) / len(tokens_a | tokens_b), 3)

    def _build_candidate_frame(
        self,
        user_id: Optional[Union[str, int]],
        conversation_type: Optional[Union[str, int]],
        business_id: Optional[Union[str, int]] = None,
        group_id: Optional[Union[str, int]] = None,
        message_text: Optional[Union[str, Any]] = None,
        media_type: Optional[Union[str, int]] = None,
        forwarded_count: Optional[Union[int, float]] = None,
    ) -> pd.DataFrame:
        """Create a candidate DataFrame ranked by lightweight rule-based similarity."""
        candidates = self.history.copy()

        if user_id is not None and not pd.isna(user_id):
            candidates = candidates[candidates["user_id"].astype(str) == str(user_id)]

        if conversation_type is not None and not pd.isna(conversation_type):
            candidates = candidates[candidates["conversation_type"].astype(str) == str(conversation_type)]

        if candidates.empty:
            return pd.DataFrame(columns=self.history.columns)

        if business_id is not None and not pd.isna(business_id):
            candidates["business_match"] = candidates["business_id"].astype(str) == str(business_id)
        else:
            candidates["business_match"] = False

        if group_id is not None and not pd.isna(group_id):
            candidates["group_match"] = candidates["group_id"].astype(str) == str(group_id)
        else:
            candidates["group_match"] = False

        if media_type is not None and not pd.isna(media_type):
            candidates["media_match"] = candidates["media_type"].astype(str) == str(media_type)
        else:
            candidates["media_match"] = False

        if forwarded_count is not None and not pd.isna(forwarded_count):
            candidates["forwarded_match"] = candidates["forwarded_count"].astype(float) == float(forwarded_count)
        else:
            candidates["forwarded_match"] = False

        candidates["text_similarity"] = candidates["message_text"].apply(
            lambda value: self._token_overlap_score(value, message_text)
        )

        candidates["score"] = (
            3.0 * candidates["business_match"].astype(float)
            + 2.0 * candidates["group_match"].astype(float)
            + 1.5 * candidates["media_match"].astype(float)
            + 1.0 * candidates["forwarded_match"].astype(float)
            + candidates["text_similarity"]
        )

        return candidates.sort_values(["score", "text_similarity"], ascending=False).reset_index(drop=True)

    def get_evidence_messages(
        self,
        user_id: Optional[Union[str, int]],
        conversation_type: Optional[Union[str, int]] = None,
        business_id: Optional[Union[str, int]] = None,
        group_id: Optional[Union[str, int]] = None,
        message_text: Optional[Union[str, Any]] = None,
        media_type: Optional[Union[str, int]] = None,
        forwarded_count: Optional[Union[int, float]] = None,
        max_results: int = 3,
    ) -> List[Dict[str, Any]]:
        """Return the best historical evidence messages using rule-based scoring.

        The retrieval algorithm first filters by the user and conversation type, then
        re-ranks candidates by matching business, group, media, forwarded count, and
        text token overlap. This is deterministic and uses no machine learning.
        """
        candidates = self._build_candidate_frame(
            user_id=user_id,
            conversation_type=conversation_type,
            business_id=business_id,
            group_id=group_id,
            message_text=message_text,
            media_type=media_type,
            forwarded_count=forwarded_count,
        )

        if candidates.empty:
            return []

        top_candidates = candidates.head(max_results)
        return [
            {
                "message_id": row["message_id"],
                "score": round(float(row["score"]), 3),
                "conversation_type": row["conversation_type"],
                "business_id": row["business_id"],
                "group_id": row["group_id"],
                "media_type": row["media_type"],
                "forwarded_count": row["forwarded_count"],
                "message_text": row["message_text"],
            }
            for _, row in top_candidates.iterrows()
        ]

    def find_similar_messages(
        self,
        user_id: Optional[Union[str, int]],
        conversation_type: Optional[Union[str, int]] = None,
        business_id: Optional[Union[str, int]] = None,
        group_id: Optional[Union[str, int]] = None,
        message_text: Optional[Union[str, Any]] = None,
        media_type: Optional[Union[str, int]] = None,
        forwarded_count: Optional[Union[int, float]] = None,
        max_results: int = 3,
    ) -> List[str]:
        """Return the best historical message ids for evidence retrieval."""
        evidence = self.get_evidence_messages(
            user_id=user_id,
            conversation_type=conversation_type,
            business_id=business_id,
            group_id=group_id,
            message_text=message_text,
            media_type=media_type,
            forwarded_count=forwarded_count,
            max_results=max_results,
        )
        return [item["message_id"] for item in evidence]

    def get_evidence_message_ids(
        self,
        user_id: Optional[Union[str, int]],
        conversation_type: Optional[Union[str, int]] = None,
        business_id: Optional[Union[str, int]] = None,
        group_id: Optional[Union[str, int]] = None,
        message_text: Optional[Union[str, Any]] = None,
        media_type: Optional[Union[str, int]] = None,
        forwarded_count: Optional[Union[int, float]] = None,
        max_results: int = 3,
    ) -> List[str]:
        """Convenience wrapper returning only evidence message ids."""
        return self.find_similar_messages(
            user_id=user_id,
            conversation_type=conversation_type,
            business_id=business_id,
            group_id=group_id,
            message_text=message_text,
            media_type=media_type,
            forwarded_count=forwarded_count,
            max_results=max_results,
        )

    def get_user_events(self, user_id: Optional[Union[str, int]]) -> pd.DataFrame:
        """Return the event rows for a user, if present."""
        if user_id is None or pd.isna(user_id):
            return pd.DataFrame(columns=self.events.columns)

        return self.events[self.events["user_id"].astype(str) == str(user_id)].reset_index(drop=True)

    def calculate_engagement(self, user_id: Optional[Union[str, int]]) -> float:
        """Calculate a simple engagement score between 0 and 1 for a user."""
        events = self.get_user_events(user_id)
        if events.empty:
            return 0.0

        opened_rate = events["message_opened"].mean()
        reply_rate = events["message_replied"].mean()
        dismissal_rate = events["notification_dismissed"].mean()

        engagement = 0.4 * opened_rate + 0.4 * reply_rate + 0.2 * (1 - dismissal_rate)
        return round(float(engagement), 3)

    def calculate_reply_rate(self, user_id: Optional[Union[str, int]]) -> float:
        """Calculate the fraction of messages that were replied to by the user."""
        events = self.get_user_events(user_id)
        if events.empty:
            return 0.0
        return round(float(events["message_replied"].mean()), 3)

    def calculate_dismissal_rate(self, user_id: Optional[Union[str, int]]) -> float:
        """Calculate the fraction of messages that were dismissed."""
        events = self.get_user_events(user_id)
        if events.empty:
            return 0.0
        return round(float(events["notification_dismissed"].mean()), 3)

    def calculate_report_rate(self, user_id: Optional[Union[str, int]]) -> float:
        """Calculate the fraction of messages that were reported."""
        events = self.get_user_events(user_id)
        if events.empty:
            return 0.0
        return round(float(events["message_reported"].mean()), 3)

    def calculate_mute_probability(self, user_id: Optional[Union[str, int]]) -> float:
        """Estimate the likelihood that the user mutes future messages."""
        events = self.get_user_events(user_id)
        if events.empty:
            return 0.0
        return round(float(events["muted_after_message"].mean()), 3)