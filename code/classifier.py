from __future__ import annotations

from typing import Dict, List, Tuple


class ClassificationResult:
    """Structured classification output with a label and confidence score."""

    def __init__(self, message_type: str, confidence: float, reasons: List[str]) -> None:
        self.message_type = message_type
        self.confidence = confidence
        self.reasons = reasons

    def to_dict(self) -> Dict[str, object]:
        """Return the result as a plain dictionary."""
        return {
            "message_type": self.message_type,
            "confidence": self.confidence,
            "reasons": self.reasons,
        }


def classify_message(features: Dict[str, object]) -> Tuple[str, float, List[str]]:
    """Classify a message into one of the supported message types.

    The classifier is rule-based and uses the extracted feature dictionary. Each
    candidate label is scored from the available boolean signals, and the highest
    scoring label is returned with a confidence value and reason list.
    """
    normalized_features = {key: bool(value) for key, value in features.items()}

    scores: Dict[str, float] = {
        "personal": 0.0,
        "urgent": 0.0,
        "event": 0.0,
        "payment": 0.0,
        "business_update": 0.0,
        "promotion": 0.0,
        "greeting": 0.0,
        "forward": 0.0,
        "spam": 0.0,
        "scam": 0.0,
        "unknown": 0.0,
    }

    reasons: Dict[str, List[str]] = {label: [] for label in scores}

    if normalized_features.get("scam", False):
        scores["scam"] += 2.5
        reasons["scam"].append("matched scam-related indicators")
    if normalized_features.get("contains_otp", False):
        scores["scam"] += 1.0
        reasons["scam"].append("contains OTP language")
    if normalized_features.get("contains_phone_number", False):
        scores["scam"] += 0.8
        reasons["scam"].append("contains a phone number")

    if normalized_features.get("payment", False):
        scores["payment"] += 2.0
        reasons["payment"].append("matched payment or billing language")
    if normalized_features.get("contains_amount", False):
        scores["payment"] += 0.7
        reasons["payment"].append("contains an amount")

    if normalized_features.get("promotion", False):
        scores["promotion"] += 2.0
        reasons["promotion"].append("matched promotional language")
    if normalized_features.get("contains_deadline", False):
        scores["promotion"] += 0.4
        reasons["promotion"].append("contains deadline-style urgency")

    if normalized_features.get("event", False):
        scores["event"] += 2.0
        reasons["event"].append("matched event or meeting language")
    if normalized_features.get("contains_meeting", False):
        scores["event"] += 0.8
        reasons["event"].append("contains meeting or scheduling language")

    if normalized_features.get("urgency", False):
        scores["urgent"] += 2.0
        reasons["urgent"].append("matched urgency indicators")
    if normalized_features.get("contains_deadline", False):
        scores["urgent"] += 0.8
        reasons["urgent"].append("contains deadline language")

    if normalized_features.get("business_update", False):
        scores["business_update"] += 2.0
        reasons["business_update"].append("matched business update language")
    if normalized_features.get("contains_link", False):
        scores["business_update"] += 0.4
        reasons["business_update"].append("contains a link")

    if normalized_features.get("greeting", False):
        scores["greeting"] += 1.8
        reasons["greeting"].append("matched greeting language")

    if normalized_features.get("forwarded_likelihood", False):
        scores["forward"] += 2.0
        reasons["forward"].append("appears to be forwarded")
    if normalized_features.get("contains_link", False):
        scores["forward"] += 0.3
        reasons["forward"].append("contains a link")

    if normalized_features.get("spam", False):
        scores["spam"] += 2.0
        reasons["spam"].append("matched spam-like language")
    if normalized_features.get("contains_link", False):
        scores["spam"] += 0.4
        reasons["spam"].append("contains a link")

    if not any(scores.values()):
        scores["unknown"] += 1.0
        reasons["unknown"].append("did not match strong rule-based indicators")

    best_label = max(scores, key=scores.get)
    best_score = max(scores.values())

    confidence = round(min(0.99, max(0.1, best_score / 3.5)), 3)
    if best_label == "unknown":
        confidence = 0.2

    return best_label, confidence, reasons[best_label]