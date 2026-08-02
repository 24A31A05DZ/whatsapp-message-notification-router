from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional, Sequence, Union

import pandas as pd


class OutputWriter:
    """Write validated prediction rows to dataset/output.csv."""

    REQUIRED_COLUMNS = [
        "message_id",
        "action",
        "message_type",
        "reason",
        "confidence",
        "evidence_message_ids",
    ]

    def __init__(self, output_path: Optional[Union[str, Path]] = None) -> None:
        self.rows: List[dict[str, Any]] = []
        self.output_path = self._resolve_output_path(output_path)

    def _resolve_output_path(self, output_path: Optional[Union[str, Path]]) -> Path:
        """Resolve the output CSV path relative to the repository root."""
        if output_path is None:
            base_dir = Path(__file__).resolve().parents[1]
            return base_dir / "dataset" / "output.csv"

        path = Path(output_path)
        if not path.is_absolute():
            base_dir = Path(__file__).resolve().parents[1]
            path = base_dir / path

        return path.resolve()

    def _validate_row(self, row: dict[str, Any]) -> None:
        """Ensure the row contains the required schema and valid values."""
        missing_columns = [column for column in self.REQUIRED_COLUMNS if column not in row]
        if missing_columns:
            raise ValueError(f"Prediction row is missing required columns: {', '.join(missing_columns)}")

        confidence = row.get("confidence")
        try:
            numeric_confidence = float(confidence)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid confidence value: {confidence}") from exc

        if not 0.0 <= numeric_confidence <= 1.0:
            raise ValueError(f"Confidence must be between 0 and 1 inclusive, got {confidence}")

    def add(
        self,
        message_id: Any,
        action: Any,
        message_type: Any,
        reason: Any,
        confidence: Any,
        evidence: Sequence[Any],
    ) -> None:
        """Add a prediction row after validating it."""
        evidence_items = [str(item) for item in evidence or []]
        normalized_evidence = "none" if not evidence_items else ";".join(evidence_items)

        row = {
            "message_id": message_id,
            "action": action,
            "message_type": message_type,
            "reason": reason,
            "confidence": float(confidence),
            "evidence_message_ids": normalized_evidence,
        }

        self._validate_row(row)
        self.rows.append(row)

    def _validate_dataframe(self, dataframe: pd.DataFrame) -> None:
        """Ensure the dataframe contains the output schema before writing."""
        missing_columns = [column for column in self.REQUIRED_COLUMNS if column not in dataframe.columns]
        if missing_columns:
            raise ValueError(f"Output data is missing required columns: {', '.join(missing_columns)}")

    def save(self) -> None:
        """Write the collected predictions to dataset/output.csv."""
        dataframe = pd.DataFrame(self.rows)
        self._validate_dataframe(dataframe)

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        dataframe.to_csv(self.output_path, index=False)
        print("Saved", len(dataframe), "predictions")