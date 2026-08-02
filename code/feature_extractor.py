import re
from typing import Dict, Optional

URGENT_WORDS = (
    "urgent",
    "immediately",
    "asap",
    "now",
    "deadline",
    "important",
    "today",
)

PAYMENT_WORDS = (
    "payment",
    "invoice",
    "bill",
    "due",
    "upi",
    "bank",
    "refund",
    "pay",
)

BUSINESS_UPDATE_WORDS = (
    "update",
    "announcement",
    "business",
    "company",
    "report",
    "policy",
)

EVENT_WORDS = (
    "meeting",
    "event",
    "birthday",
    "wedding",
    "conference",
    "party",
)

GREETING_WORDS = ("hi", "hello", "hey", "good morning", "good evening")

PROMOTION_WORDS = (
    "sale",
    "offer",
    "discount",
    "coupon",
    "deal",
    "free",
    "promotion",
)

SCAM_WORDS = (
    "lottery",
    "claim reward",
    "kyc",
    "click here",
    "win",
    "congratulations",
    "prize",
)

SPAM_WORDS = (
    "spam",
    "bulk",
    "broadcast",
    "marketing",
    "subscribe",
    "unsubscribe",
)

FORWARDED_WORDS = ("forwarded", "fwd", "forward")

LINK_PATTERN = re.compile(r"https?://|www\.", re.IGNORECASE)
PHONE_PATTERN = re.compile(r"(?:\+?\d[\d -]{8,}\d)")
OTP_PATTERN = re.compile(r"\b(?:otp|one time password|one-time password)\b", re.IGNORECASE)
AMOUNT_PATTERN = re.compile(r"(?:rs|inr|₹|\$|€)?\s?\d+(?:[.,]\d{1,2})?(?:\s?(?:k|m|l|cr|lac))?", re.IGNORECASE)
DEADLINE_PATTERN = re.compile(r"\b(?:by|before|until|deadline|due|tomorrow|today)\b", re.IGNORECASE)
MEETING_PATTERN = re.compile(r"\b(?:meeting|call|zoom|teams|schedule|join)\b", re.IGNORECASE)
REMINDER_PATTERN = re.compile(r"\b(?:reminder|remember|note|follow up|follow-up)\b", re.IGNORECASE)
IMAGE_PATTERN = re.compile(r"\b(image|photo|screenshot|picture|media)\b", re.IGNORECASE)
VOICE_PATTERN = re.compile(r"\b(audio|voice note|voicenote|voice message|recording)\b", re.IGNORECASE)


def _contains_any(text: str, words: tuple[str, ...]) -> bool:
    """Return True when any of the provided words appear in the text."""
    return any(word in text for word in words)


def _contains_pattern(text: str, pattern: re.Pattern[str]) -> bool:
    """Return True when the provided regex pattern matches the text."""
    return bool(pattern.search(text))


def extract_features(message_text: Optional[str]) -> Dict[str, bool]:
    """Extract deterministic, rule-based features from a message string.

    The returned dictionary is flat and reusable for routing logic. Each key is a
    boolean indicator for a specific message trait:
    - urgency: signals that the message requires immediate attention.
    - payment: indicates a financial request or transaction-related message.
    - business_update: suggests an organizational or company announcement.
    - event: points to a calendar or social event.
    - greeting: suggests a friendly opener or hello.
    - promotion: indicates a sale, offer, or discount.
    - scam: flags suspicious or fraud-like content.
    - spam: points to broadcast or mass-marketing content.
    - forwarded_likelihood: suggests the message was forwarded.
    - contains_link: detects URL-style links.
    - contains_phone_number: detects a phone number pattern.
    - contains_otp: detects one-time-password or OTP language.
    - contains_amount: detects money values or amounts.
    - contains_deadline: detects time-sensitive or deadline language.
    - contains_meeting: detects scheduling or meeting language.
    - contains_reminder: detects reminder or follow-up language.
    - contains_image_message: detects image or screenshot-related language.
    - contains_voice_message: detects voice-note or audio-related language.
    """
    text = str(message_text or "").strip().lower()

    features: Dict[str, bool] = {
        "urgency": _contains_any(text, URGENT_WORDS),
        "payment": _contains_any(text, PAYMENT_WORDS),
        "business_update": _contains_any(text, BUSINESS_UPDATE_WORDS),
        "event": _contains_any(text, EVENT_WORDS),
        "greeting": _contains_any(text, GREETING_WORDS),
        "promotion": _contains_any(text, PROMOTION_WORDS),
        "scam": _contains_any(text, SCAM_WORDS),
        "spam": _contains_any(text, SPAM_WORDS),
        "forwarded_likelihood": _contains_any(text, FORWARDED_WORDS),
        "contains_link": _contains_pattern(text, LINK_PATTERN),
        "contains_phone_number": _contains_pattern(text, PHONE_PATTERN),
        "contains_otp": _contains_pattern(text, OTP_PATTERN),
        "contains_amount": _contains_pattern(text, AMOUNT_PATTERN),
        "contains_deadline": _contains_pattern(text, DEADLINE_PATTERN),
        "contains_meeting": _contains_pattern(text, MEETING_PATTERN),
        "contains_reminder": _contains_pattern(text, REMINDER_PATTERN),
        "contains_image_message": _contains_pattern(text, IMAGE_PATTERN),
        "contains_voice_message": _contains_pattern(text, VOICE_PATTERN),
    }

    # Preserve the older field names expected by the existing classifier.
    features.update(
        {
            "urgent": features["urgency"],
            "contains_link": features["contains_link"],
        }
    )

    return features