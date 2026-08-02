from __future__ import annotations

from typing import Any, Dict, List, Optional


def _feature_value(features: Dict[str, Any], *keys: str) -> bool:
    """Return True when any of the requested feature keys is present and truthy."""
    for key in keys:
        if key in features and bool(features[key]):
            return True
    return False


def _coerce_business_context(business_context: Any, business_verified: bool) -> Dict[str, Any]:
    """Normalize business context into a small dictionary for routing."""
    if isinstance(business_context, dict):
        return {
            "verified": bool(business_context.get("verified", business_verified)),
            "trust_score": float(business_context.get("trust_score", 50.0)),
            "risk_score": float(business_context.get("risk_score", 0.0)),
            "reports": int(business_context.get("reports", 0)),
        }

    if business_context is None:
        return {"verified": bool(business_verified), "trust_score": 50.0, "risk_score": 0.0, "reports": 0}

    return {
        "verified": bool(getattr(business_context, "verified", business_verified)),
        "trust_score": float(getattr(business_context, "trust_score", 50.0)),
        "risk_score": float(getattr(business_context, "risk_score", 0.0)),
        "reports": int(getattr(business_context, "reports", 0)),
    }


def _coerce_user_profile(user_profile: Any) -> Dict[str, Any]:
    """Normalize the user profile into a small dictionary for routing."""
    if user_profile is None:
        return {"dismiss_rate": 0.0, "reply_rate": 0.0, "open_rate": 0.0, "current_notification_load": 0, "preferences": {}}

    if isinstance(user_profile, dict):
        preferences = user_profile.get("preferences", {}) or {}
        return {
            "dismiss_rate": float(user_profile.get("dismiss_rate", 0.0)),
            "reply_rate": float(user_profile.get("reply_rate", 0.0)),
            "open_rate": float(user_profile.get("open_rate", 0.0)),
            "current_notification_load": int(user_profile.get("current_notification_load", 0)),
            "preferences": preferences,
        }

    return {
        "dismiss_rate": float(getattr(user_profile, "dismiss_rate", 0.0)),
        "reply_rate": float(getattr(user_profile, "reply_rate", 0.0)),
        "open_rate": float(getattr(user_profile, "open_rate", 0.0)),
        "current_notification_load": int(getattr(user_profile, "current_notification_load", 0)),
        "preferences": getattr(user_profile, "preferences", {}) or {},
    }


def _coerce_group_context(group_context: Any) -> Dict[str, Any]:
    """Normalize group context into a small dictionary for routing."""
    if group_context is None:
        return {"muted_group": False, "admin_messages": False, "high_priority_group": False, "family_group": False, "school_group": False, "society_group": False, "user_engagement": 0.0}

    if isinstance(group_context, dict):
        return {
            "muted_group": bool(group_context.get("muted_group", False)),
            "admin_messages": bool(group_context.get("admin_messages", False)),
            "high_priority_group": bool(group_context.get("high_priority_group", False)),
            "family_group": bool(group_context.get("family_group", False)),
            "school_group": bool(group_context.get("school_group", False)),
            "society_group": bool(group_context.get("society_group", False)),
            "user_engagement": float(group_context.get("user_engagement", 0.0)),
        }

    return {
        "muted_group": bool(getattr(group_context, "muted_group", False)),
        "admin_messages": bool(getattr(group_context, "admin_messages", False)),
        "high_priority_group": bool(getattr(group_context, "high_priority_group", False)),
        "family_group": bool(getattr(group_context, "family_group", False)),
        "school_group": bool(getattr(group_context, "school_group", False)),
        "society_group": bool(getattr(group_context, "society_group", False)),
        "user_engagement": float(getattr(group_context, "user_engagement", 0.0)),
    }


def _coerce_history_context(history_context: Any) -> Dict[str, Any]:
    """Normalize history context into a small dictionary for routing."""
    if history_context is None:
        return {"engagement": 0.0, "reply_rate": 0.0, "dismissal_rate": 0.0, "report_rate": 0.0, "mute_probability": 0.0, "evidence_message_ids": []}

    if isinstance(history_context, dict):
        return {
            "engagement": float(history_context.get("engagement", 0.0)),
            "reply_rate": float(history_context.get("reply_rate", 0.0)),
            "dismissal_rate": float(history_context.get("dismissal_rate", 0.0)),
            "report_rate": float(history_context.get("report_rate", 0.0)),
            "mute_probability": float(history_context.get("mute_probability", 0.0)),
            "evidence_message_ids": list(history_context.get("evidence_message_ids", [])),
        }

    return {
        "engagement": float(getattr(history_context, "engagement", 0.0)),
        "reply_rate": float(getattr(history_context, "reply_rate", 0.0)),
        "dismissal_rate": float(getattr(history_context, "dismissal_rate", 0.0)),
        "report_rate": float(getattr(history_context, "report_rate", 0.0)),
        "mute_probability": float(getattr(history_context, "mute_probability", 0.0)),
        "evidence_message_ids": list(getattr(history_context, "evidence_message_ids", []) or []),
    }


def route_message(
    message_type: str,
    features: Dict[str, Any],
    business_context: Any = None,
    user_profile: Any = None,
    group_context: Any = None,
    history_context: Any = None,
    media_content: Optional[str] = None,
    business_verified: bool = False,
    user_prefers_business: bool = True,
) -> Dict[str, Any]:
    """Route a message to notify, digest, or mute using rule-based evidence.

    The routing algorithm combines the detected message type, extracted features,
    user preferences, group context, business trust signals, historical evidence,
    and media content. Scores are assigned to each action and the highest-scoring
    action is selected. The confidence is derived from the difference between the
    winning and runner-up action scores, making it self-adjusting rather than
    relying on a hardcoded threshold.
    """
    business = _coerce_business_context(business_context, business_verified)
    user = _coerce_user_profile(user_profile)
    group = _coerce_group_context(group_context)
    history = _coerce_history_context(history_context)

    message_type = str(message_type or "unknown").lower()
    media_text = str(media_content or "")
    media_features = bool(media_text.strip())

    scores = {"notify": 0.0, "digest": 0.0, "mute": 0.0}
    decision_path: List[str] = []

    if message_type == "scam" or _feature_value(features, "scam"):
        scores["mute"] += 3.0
        decision_path.append("scam-like indicators increase mute likelihood")
    if _feature_value(features, "contains_otp"):
        scores["mute"] += 1.2
        decision_path.append("OTP language raises risk")
    if _feature_value(features, "contains_phone_number"):
        scores["mute"] += 0.8
        decision_path.append("phone-number pattern increases caution")
    if group.get("muted_group"):
        scores["mute"] += 1.0
        decision_path.append("the user has muted this group")
    if history.get("report_rate", 0.0) > 0.2:
        scores["mute"] += 0.8
        decision_path.append("history shows repeated reporting")
    if business.get("risk_score", 0.0) > 50.0:
        scores["mute"] += 0.6
        decision_path.append("business risk score is elevated")

    if message_type == "urgent" or _feature_value(features, "urgency"):
        scores["notify"] += 2.8
        decision_path.append("urgent wording suggests immediate attention")
    if message_type in {"event", "business_update"} or _feature_value(features, "event", "business_update"):
        scores["notify"] += 2.0
        decision_path.append("event or business-update language is time-sensitive")
    if message_type == "payment" and business.get("verified"):
        scores["notify"] += 2.0
        decision_path.append("verified business payment context is high confidence")
    elif message_type == "payment":
        scores["digest"] += 1.0
        decision_path.append("payment without verified business is less urgent")
    if group.get("high_priority_group") or group.get("admin_messages"):
        scores["notify"] += 0.9
        decision_path.append("the group context is high priority")
    if user.get("reply_rate", 0.0) > 0.3 and user.get("dismiss_rate", 0.0) < 0.4:
        scores["notify"] += 0.5
        decision_path.append("the user is engaged and likely to respond")
    if media_features:
        scores["notify"] += 0.3
        decision_path.append("media content adds context")

    if message_type in {"personal", "greeting"} or _feature_value(features, "greeting"):
        scores["digest"] += 1.6
        decision_path.append("personal or greeting content is usually low urgency")
    if message_type == "promotion" or _feature_value(features, "promotion"):
        if user_prefers_business:
            scores["digest"] += 1.6
            decision_path.append("promotions are better batched for this user")
        else:
            scores["mute"] += 0.8
            decision_path.append("promotions are less welcome for this user")
    if message_type == "forward" or _feature_value(features, "forwarded_likelihood"):
        scores["digest"] += 0.6
        decision_path.append("forwarded content is less trustworthy")
    if not any([scores["notify"], scores["digest"], scores["mute"]]):
        scores["digest"] += 1.0
        decision_path.append("no strong signal was found, so the message defaults to digest")

    if user.get("dismiss_rate", 0.0) > 0.4:
        scores["digest"] += 0.4
        decision_path.append("high dismissal rates suggest batching instead of interruption")
    if user.get("current_notification_load", 0) > 20:
        scores["digest"] += 0.3
        decision_path.append("the user currently has a heavy notification load")
    if history.get("mute_probability", 0.0) > 0.5:
        scores["mute"] += 0.7
        decision_path.append("historical mute probability is elevated")
    if history.get("engagement", 0.0) > 0.6 and user.get("dismiss_rate", 0.0) < 0.3:
        scores["notify"] += 0.4
        decision_path.append("prior engagement supports a more immediate action")

    best_action = max(scores, key=scores.get)
    second_action = sorted(scores.items(), key=lambda item: item[1], reverse=True)[1][0]
    margin = max(scores.values()) - min(scores.values())
    confidence = round(min(0.99, max(0.1, 0.45 + (margin / 8.0))), 3)

    reason = "; ".join(decision_path[:3]) if decision_path else f"{best_action} selected based on the available context"

    return {
        "action": best_action,
        "reason": reason,
        "confidence": confidence,
        "decision_path": decision_path,
        "scores": scores,
    }


def decide_action(
    message_type: str,
    features: Dict[str, Any],
    business_verified: bool = False,
    user_prefers_business: bool = True,
) -> str:
    """Backward-compatible wrapper returning the routing decision as a single action."""
    result = route_message(
        message_type=message_type,
        features=features,
        business_context=None,
        user_profile=None,
        group_context=None,
        history_context=None,
        media_content=None,
        business_verified=business_verified,
        user_prefers_business=user_prefers_business,
    )
    return str(result["action"])
