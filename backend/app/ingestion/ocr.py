"""Thin wrapper around Tesseract OCR (via pytesseract).

Isolated in its own module so both the image processor and the PDF
processor's scanned-page fallback can share one implementation, and so
Tesseract availability can be checked/mocked in one place.
"""
from __future__ import annotations

from PIL import Image

from app.config import settings
from app.core.exceptions import OCRError
from app.core.logging import get_logger

logger = get_logger(__name__)

if settings.TESSERACT_CMD:
    import pytesseract

    pytesseract.pytesseract.tesseract_cmd = settings.TESSERACT_CMD
else:
    import pytesseract


def run_ocr(image: Image.Image) -> str:
    """Run Tesseract OCR on a PIL image and return extracted text.

    Raises OCRError (never a raw exception) so callers can decide how to
    handle failure without crashing the ingestion batch.
    """
    try:
        text = pytesseract.image_to_string(image, lang=settings.OCR_LANGUAGE)
        return text
    except pytesseract.TesseractNotFoundError as exc:
        logger.error("Tesseract binary not found on PATH: %s", exc)
        raise OCRError(
            "Tesseract OCR engine is not installed or not on PATH. "
            "See README for installation instructions."
        ) from exc
    except Exception as exc:  # pragma: no cover - defensive catch-all
        logger.error("OCR failed: %s", exc)
        raise OCRError(f"OCR processing failed: {exc}") from exc


def is_tesseract_available() -> bool:
    try:
        pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False
