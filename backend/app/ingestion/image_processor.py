"""Image (PNG/JPG/JPEG) text extraction via Tesseract OCR.

Images are always treated as a single content unit with page_number=1
and ocr_used=True, since all text from a raster image is necessarily
OCR-derived.
"""
from __future__ import annotations

from pathlib import Path
from typing import List

from PIL import Image, UnidentifiedImageError

from app.core.exceptions import DocumentProcessingError, OCRError
from app.core.logging import get_logger
from app.ingestion.base import ExtractedUnit
from app.ingestion.normalization import is_effectively_empty, normalize_text
from app.ingestion.ocr import run_ocr

logger = get_logger(__name__)


def extract_image(file_path: Path) -> List[ExtractedUnit]:
    try:
        image = Image.open(file_path)
        image.load()
    except UnidentifiedImageError as exc:
        raise DocumentProcessingError(f"Not a valid image file: {exc}") from exc
    except Exception as exc:
        raise DocumentProcessingError(f"Could not open image: {exc}") from exc

    try:
        raw_text = run_ocr(image)
    except OCRError as exc:
        # Do not crash the backend — surface as a processing error for
        # this document only.
        logger.error("OCR failed for image %s: %s", file_path.name, exc.message)
        raise DocumentProcessingError(f"OCR failed: {exc.message}") from exc

    normalized = normalize_text(raw_text)
    if is_effectively_empty(normalized):
        logger.info("No text detected via OCR in image %s", file_path.name)
        return []

    return [ExtractedUnit(text=normalized, page_number=1, ocr_used=True)]
