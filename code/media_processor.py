from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import pandas as pd


class MediaProcessor:
    """Isolate image OCR and voice transcription for multimodal messages."""

    def __init__(self, data_dir: Optional[Union[str, Path]] = None) -> None:
        self._data_dir = self._resolve_data_dir(data_dir)
        self.images = self._load_images()
        self.voice_notes = self._load_voice_notes()

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

    def _load_images(self) -> pd.DataFrame:
        """Load image metadata from disk."""
        file_path = self._data_dir / "images.csv"
        if not file_path.exists():
            raise FileNotFoundError(f"Images dataset not found: {file_path}")
        return pd.read_csv(file_path)

    def _load_voice_notes(self) -> pd.DataFrame:
        """Load voice note metadata from disk."""
        file_path = self._data_dir / "voice_notes.csv"
        if not file_path.exists():
            raise FileNotFoundError(f"Voice notes dataset not found: {file_path}")
        return pd.read_csv(file_path)

    def _find_image_path(self, image_id: Optional[Union[str, int]]) -> Optional[Path]:
        """Resolve the image file path for an image id."""
        if image_id is None or pd.isna(image_id):
            return None

        matches = self.images[self.images["image_id"] == str(image_id)]
        if matches.empty:
            return None

        relative_path = str(matches.iloc[0].get("file_path", ""))
        if not relative_path:
            return None
        return (self._data_dir / relative_path).resolve()

    def _find_voice_path(self, voice_note_id: Optional[Union[str, int]]) -> Optional[Path]:
        """Resolve the audio file path for a voice note id."""
        if voice_note_id is None or pd.isna(voice_note_id):
            return None

        matches = self.voice_notes[self.voice_notes["voice_note_id"] == str(voice_note_id)]
        if matches.empty:
            return None

        relative_path = str(matches.iloc[0].get("file_path", ""))
        if not relative_path:
            return None
        return (self._data_dir / relative_path).resolve()

    def extract_image_text(self, image_id: Optional[Union[str, int]]) -> str:
        """Return OCR-like text for an image when the file is available.

        The current environment may not have OCR libraries installed, so this method
        uses a simple abstraction: if the image cannot be read by an OCR engine, it
        returns a descriptive placeholder rather than failing the pipeline.
        """
        image_path = self._find_image_path(image_id)
        if image_path is None or not image_path.exists():
            return ""

        try:
            from PIL import Image  # type: ignore
        except ImportError:
            return f"[image placeholder] {image_path.name}"

        try:
            image = Image.open(image_path)
            image.load()
        except Exception:
            return f"[image placeholder] {image_path.name}"

        try:
            import pytesseract  # type: ignore
        except ImportError:
            return f"[image placeholder] {image_path.name}"

        try:
            text = pytesseract.image_to_string(image)
        except Exception:
            return f"[image placeholder] {image_path.name}"

        return text.strip() or f"[image placeholder] {image_path.name}"

    def transcribe_voice_note(self, voice_note_id: Optional[Union[str, int]]) -> str:
        """Transcribe a voice note using Whisper if available.

        If Whisper is not installed, the method returns a placeholder transcript so
        the rest of the system can remain isolated and deterministic.
        """
        voice_path = self._find_voice_path(voice_note_id)
        if voice_path is None or not voice_path.exists():
            return ""

        try:
            import whisper  # type: ignore
        except ImportError:
            return f"[voice placeholder] {voice_path.name}"

        try:
            model = whisper.load_model("base")
            result = model.transcribe(str(voice_path), fp16=False)
        except Exception:
            return f"[voice placeholder] {voice_path.name}"

        text = result.get("text", "").strip()
        return text or f"[voice placeholder] {voice_path.name}"

    def process_media(self, media_type: Optional[str], media_id: Optional[Union[str, int]]) -> str:
        """Process either an image or voice note and return extracted text or transcript."""
        if media_type is None or pd.isna(media_type):
            return ""

        normalized_type = str(media_type).strip().lower()
        if normalized_type == "image":
            return self.extract_image_text(media_id)
        if normalized_type == "voice":
            return self.transcribe_voice_note(media_id)

        return ""
