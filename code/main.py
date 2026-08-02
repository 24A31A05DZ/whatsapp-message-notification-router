from business import BusinessService
from classifier import classify_message
from feature_extractor import extract_features
from group import GroupService
from history import HistoryService
from loader import DataLoader
from media_processor import MediaProcessor
from output_writer import OutputWriter
from router import route_message
from scorer import ConfidenceScorer
from user import UserService

loader = DataLoader()
business = BusinessService()
history = HistoryService()
user_service = UserService()
group_service = GroupService()
media_processor = MediaProcessor()
writer = OutputWriter()
scorer = ConfidenceScorer()

for _, message in loader.messages.iterrows():
    text = str(message.get("message_text", ""))
    features = extract_features(text)
    message_type, classification_confidence, reasons = classify_message(features)
    user_profile = user_service.get_profile(message.get("user_id"))
    group_context = group_service.get_context(message.get("user_id"), message.get("group_id"))
    business_profile = business.get_profile(message.get("user_id"), message.get("business_id"))
    media_content = media_processor.process_media(message.get("media_type"), message.get("media_id"))
    history_context = {
        "engagement": history.calculate_engagement(message.get("user_id")),
        "reply_rate": history.calculate_reply_rate(message.get("user_id")),
        "dismissal_rate": history.calculate_dismissal_rate(message.get("user_id")),
        "report_rate": history.calculate_report_rate(message.get("user_id")),
        "mute_probability": history.calculate_mute_probability(message.get("user_id")),
        "evidence_message_ids": history.get_evidence_message_ids(
            message.get("user_id"),
            message.get("conversation_type"),
            message.get("business_id"),
            message.get("group_id"),
            text,
            message.get("media_type"),
            message.get("forwarded_count"),
        ),
    }
    decision = route_message(
        message_type,
        features,
        business_profile.to_dict(),
        user_profile.to_dict(),
        group_context.to_dict(),
        history_context,
        media_content,
        business_profile.verified,
        True,
    )
    confidence = scorer.score_confidence(
        business_profile.to_dict(),
        history_context,
        classification_confidence,
        user_profile.to_dict(),
        media_content,
    )
    writer.add(
        message_id=message.get("message_id"),
        action=decision["action"],
        message_type=message_type,
        reason=decision["reason"] + f" | {'; '.join(reasons)}",
        confidence=confidence,
        evidence=history_context["evidence_message_ids"],
    )

writer.save()