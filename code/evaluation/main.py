from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List
from difflib import SequenceMatcher

import pandas as pd


REQUIRED_COLUMNS = ["message_id", "action", "message_type", "reason", "confidence", "evidence_message_ids"]
SAMPLE_COLUMNS = ["message_id", "action", "message_type"]
MESSAGE_COLUMNS = ["message_id", "user_id", "conversation_type", "group_id", "business_id", "sender_user_id", "message_text"]


def _resolve_path(path: str | Path) -> Path:
    """Resolve a repository-relative path for the evaluation inputs and outputs."""
    if isinstance(path, Path):
        return path

    candidate = Path(path)
    if candidate.is_absolute():
        return candidate

    repo_root = Path(__file__).resolve().parents[2]
    return (repo_root / candidate).resolve()


def _load_predictions(predictions_path: Path) -> pd.DataFrame:
    """Load and validate prediction rows from the generated output file."""
    predictions = pd.read_csv(predictions_path)
    missing = [column for column in REQUIRED_COLUMNS if column not in predictions.columns]
    if missing:
        raise ValueError(f"Predictions are missing required columns: {', '.join(missing)}")
    return predictions


def _load_sample_messages(sample_path: Path) -> pd.DataFrame:
    """Load the reference sample messages and validate the expected columns."""
    sample = pd.read_csv(sample_path)
    missing = [column for column in SAMPLE_COLUMNS if column not in sample.columns]
    if missing:
        raise ValueError(f"Sample messages are missing required columns: {', '.join(missing)}")
    return sample


def _load_message_context(messages_path: Path) -> pd.DataFrame:
    """Load message metadata needed to align the sample labels to generated predictions."""
    messages = pd.read_csv(messages_path)
    missing = [column for column in MESSAGE_COLUMNS if column not in messages.columns]
    if missing:
        raise ValueError(f"Message context file is missing required columns: {', '.join(missing)}")
    return messages


def _normalize_text(text: object) -> str:
    """Normalize text for robust matching across punctuation and whitespace differences."""
    if pd.isna(text):
        return ""
    text = str(text).lower()
    for char in ("\n", "\r", "\t"):
        text = text.replace(char, " ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def _context_score(sample_row: pd.Series, prediction_row: pd.Series) -> float:
    """Score how well a sample row matches a prediction row using content and metadata."""
    text_similarity = SequenceMatcher(
        None,
        _normalize_text(sample_row.get("message_text", "")),
        _normalize_text(prediction_row.get("message_text", "")),
    ).ratio()

    checks = 0
    matches = 0
    for field in ("user_id", "conversation_type"):
        checks += 1
        matches += int(str(sample_row.get(field, "")).strip() == str(prediction_row.get(field, "")).strip())

    for field in ("group_id", "business_id", "sender_user_id"):
        sample_value = sample_row.get(field)
        prediction_value = prediction_row.get(field)
        if pd.isna(sample_value) and pd.isna(prediction_value):
            continue
        checks += 1
        if pd.isna(sample_value) or pd.isna(prediction_value):
            matches += 0
        else:
            matches += int(str(sample_value).strip() == str(prediction_value).strip())

    context_ratio = matches / checks if checks else 0.0
    return 0.7 * text_similarity + 0.3 * context_ratio


def _merge_predictions(predictions: pd.DataFrame, sample: pd.DataFrame, messages: pd.DataFrame) -> pd.DataFrame:
    """Join predictions with the sample labels using direct ids when possible, otherwise fuzzy matching."""
    direct = predictions.merge(sample, on="message_id", how="inner", suffixes=("_pred", "_sample"))
    if not direct.empty:
        return direct

    predictions_with_context = predictions.merge(
        messages[["message_id", "user_id", "conversation_type", "group_id", "business_id", "sender_user_id", "message_text"]],
        on="message_id",
        how="left",
    )

    matched_rows: List[Dict[str, object]] = []
    used_prediction_ids = set()

    for _, sample_row in sample.iterrows():
        best_match = None
        best_score = 0.0

        for _, prediction_row in predictions_with_context.iterrows():
            prediction_id = str(prediction_row.get("message_id", ""))
            if prediction_id in used_prediction_ids:
                continue

            score = _context_score(sample_row, prediction_row)
            if score >= 0.55 and score > best_score:
                best_score = score
                best_match = prediction_row

        if best_match is None:
            continue

        matched_rows.append(
            {
                "message_id": str(best_match.get("message_id", "")),
                "action_pred": best_match.get("action", ""),
                "message_type_pred": best_match.get("message_type", ""),
                "reason_pred": best_match.get("reason", ""),
                "confidence_pred": best_match.get("confidence", ""),
                "evidence_message_ids_pred": best_match.get("evidence_message_ids", ""),
                "action_sample": sample_row.get("action", ""),
                "message_type_sample": sample_row.get("message_type", ""),
                "reason_sample": sample_row.get("reason", ""),
                "confidence_sample": sample_row.get("confidence", ""),
                "evidence_message_ids_sample": sample_row.get("evidence_message_ids", ""),
                "match_score": best_score,
            }
        )
        used_prediction_ids.add(str(best_match.get("message_id", "")))

    return pd.DataFrame(matched_rows)


def _action_accuracy(merged: pd.DataFrame) -> float:
    """Measure how often the predicted action matches the sample action."""
    if merged.empty:
        return 0.0
    return round(float((merged["action_pred"].astype(str).str.strip() == merged["action_sample"].astype(str).str.strip()).mean()), 3)


def _message_type_accuracy(merged: pd.DataFrame) -> float:
    """Measure how often the predicted message type matches the sample message type."""
    if merged.empty:
        return 0.0
    return round(float((merged["message_type_pred"].astype(str).str.strip() == merged["message_type_sample"].astype(str).str.strip()).mean()), 3)


def _confidence_statistics(merged: pd.DataFrame) -> Dict[str, float]:
    """Summarize confidence values for all predictions."""
    confidences = pd.to_numeric(merged["confidence_pred"], errors="coerce").fillna(0.0)
    if confidences.empty:
        return {"mean": 0.0, "median": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": round(float(confidences.mean()), 3),
        "median": round(float(confidences.median()), 3),
        "min": round(float(confidences.min()), 3),
        "max": round(float(confidences.max()), 3),
    }


def _evidence_quality(merged: pd.DataFrame) -> Dict[str, float]:
    """Measure how often evidence is present and how much evidence is provided."""
    evidence = merged["evidence_message_ids_pred"].fillna("")
    has_evidence = evidence.astype(str).str.strip() != ""
    evidence_count = evidence.astype(str).str.split(";").str.len().clip(lower=0)
    return {
        "coverage": round(float(has_evidence.mean()), 3) if not evidence.empty else 0.0,
        "average_count": round(float(evidence_count.mean()), 3) if not evidence.empty else 0.0,
    }


def generate_report(
    predictions_path: str | Path = "dataset/output.csv",
    sample_path: str | Path = "dataset/sample_messages.csv",
    messages_path: str | Path = "dataset/messages.csv",
) -> Dict[str, object]:
    """Generate an evaluation report comparing predictions against sample messages."""
    predictions_path = _resolve_path(predictions_path)
    sample_path = _resolve_path(sample_path)
    messages_path = _resolve_path(messages_path)

    predictions = _load_predictions(predictions_path)
    sample = _load_sample_messages(sample_path)
    messages = _load_message_context(messages_path)
    merged = _merge_predictions(predictions, sample, messages)

    report = {
        "action_accuracy": _action_accuracy(merged),
        "message_type_accuracy": _message_type_accuracy(merged),
        "confidence_statistics": _confidence_statistics(merged),
        "evidence_quality": _evidence_quality(merged),
        "matched_rows": int(len(merged)),
    }

    return report


def print_report(report: Dict[str, object]) -> None:
    """Render the evaluation report in a text-friendly format."""
    print("Evaluation Report")
    print("=================")
    print(f"Matched rows: {report['matched_rows']}")
    print(f"Action Accuracy: {report['action_accuracy']} - share of actions that match the sample labels.")
    print(f"Message Type Accuracy: {report['message_type_accuracy']} - share of message types that match the sample labels.")
    print("Confidence Statistics:")
    for key, value in report["confidence_statistics"].items():
        print(f"- {key}: {value}")
    print("Evidence Quality:")
    for key, value in report["evidence_quality"].items():
        print(f"- {key}: {value}")


if __name__ == "__main__":
    report = generate_report()
    print_report(report)
