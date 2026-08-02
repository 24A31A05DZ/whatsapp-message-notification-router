from __future__ import annotations

from typing import Any, Dict, Optional


class ConfidenceScorer:
    """Combine routing signals into a normalized confidence score between 0 and 1."""

    def __init__(self) -> None:
        self._weights = {
            "business_trust": 0.25,
            "history_similarity": 0.25,
            "classification_confidence": 0.25,
            "user_engagement": 0.15,
            "media_certainty": 0.10,
        }

    def _normalize(self, value: float, lower: float = 0.0, upper: float = 1.0) -> float:
        """Clamp a numeric value into a closed [0, 1] interval."""
        if upper <= lower:
            return 0.0
        return max(lower, min(upper, value))

    def _business_trust(self, business_context: Any) -> float:
        """Compute a normalized business-trust signal from context."""
        if business_context is None:
            return 0.5

        if isinstance(business_context, dict):
            trust_score = float(business_context.get("trust_score", 50.0))
            risk_score = float(business_context.get("risk_score", 0.0))
        else:
            trust_score = float(getattr(business_context, "trust_score", 50.0))
            risk_score = float(getattr(business_context, "risk_score", 0.0))

        adjusted = trust_score / 100.0
        risk_penalty = risk_score / 100.0 * 0.4
        return self._normalize(adjusted - risk_penalty)

    def _history_similarity(self, history_context: Any) -> float:
        """Convert historical evidence quality into a normalized similarity signal."""
        if history_context is None:
            return 0.5

        if isinstance(history_context, dict):
            evidence = history_context.get("evidence_message_ids", []) or []
            engagement = float(history_context.get("engagement", 0.0))
        else:
            evidence = getattr(history_context, "evidence_message_ids", []) or []
            engagement = float(getattr(history_context, "engagement", 0.0))

        evidence_strength = 0.5 if not evidence else min(1.0, len(evidence) / 3.0)
        return self._normalize(0.5 * evidence_strength + 0.5 * engagement)

    def _classification_confidence(self, classification_confidence: Optional[float]) -> float:
        """Normalize the classifier confidence into the [0, 1] range."""
        if classification_confidence is None:
            return 0.5
        return self._normalize(float(classification_confidence))

    def _user_engagement(self, user_profile: Any) -> float:
        """Convert user engagement signals into a normalized score."""
        if user_profile is None:
            return 0.5

        if isinstance(user_profile, dict):
            dismiss_rate = float(user_profile.get("dismiss_rate", 0.0))
            reply_rate = float(user_profile.get("reply_rate", 0.0))
            open_rate = float(user_profile.get("open_rate", 0.0))
        else:
            dismiss_rate = float(getattr(user_profile, "dismiss_rate", 0.0))
            reply_rate = float(getattr(user_profile, "reply_rate", 0.0))
            open_rate = float(getattr(user_profile, "open_rate", 0.0))

        engagement_score = (0.5 * open_rate) + (0.3 * reply_rate) + (0.2 * (1.0 - dismiss_rate))
        return self._normalize(engagement_score)

    def _media_certainty(self, media_content: Optional[str]) -> float:
        """Estimate certainty from available media content."""
        if media_content is None:
            return 0.5

        text = str(media_content).strip()
        if not text:
            return 0.5

        if text.startswith("[image placeholder]") or text.startswith("[voice placeholder]"):
            return 0.3
        return 0.8

    def score_confidence(
        self,
        business_context: Any = None,
        history_context: Any = None,
        classification_confidence: Optional[float] = None,
        user_profile: Any = None,
        media_content: Optional[str] = None,
    ) -> float:
        """Calculate a weighted confidence score between 0 and 1.

        The scoring formula is:
        confidence = 0.25 * business_trust
                   + 0.25 * history_similarity
                   + 0.25 * classification_confidence
                   + 0.15 * user_engagement
                   + 0.10 * media_certainty

        Each contributing signal is normalized to the [0, 1] range before weighting.
        """
        business_value = self._business_trust(business_context)
        history_value = self._history_similarity(history_context)
        classification_value = self._classification_confidence(classification_confidence)
        user_value = self._user_engagement(user_profile)
        media_value = self._media_certainty(media_content)

        score = (
            self._weights["business_trust"] * business_value
            + self._weights["history_similarity"] * history_value
            + self._weights["classification_confidence"] * classification_value
            + self._weights["user_engagement"] * user_value
            + self._weights["media_certainty"] * media_value
        )

        return round(self._normalize(score), 3)
