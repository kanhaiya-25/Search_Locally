"""PDF text extraction using PyMuPDF (fitz), with optional OCR fallback
for scanned pages.

Flow per page:
  1. Extract native text with PyMuPDF.
  2. If the extracted text is shorter than `PDF_OCR_MIN_CHARS` and OCR is
     enabled, render the page to an image and run Tesseract on it. This
     avoids OCR-ing every page of a normal text PDF, which would be
     slow and unnecessary.
  3. Normalize whitespace.
  4. Skip pages that are still effectively empty after extraction/OCR.
"""
from __future__ import annotations

from pathlib import Path
from typing import List

import fitz  # PyMuPDF
from PIL import Image

from app.config import settings
from app.core.exceptions import DocumentProcessingError, OCRError
from app.core.logging import get_logger
from app.ingestion.base import ExtractedUnit
from app.ingestion.normalization import is_effectively_empty, normalize_text
from app.ingestion.ocr import run_ocr

logger = get_logger(__name__)

# Render scale for OCR fallback. 2.0 roughly doubles pixel density
# (~144 DPI from a 72 DPI base), which improves OCR accuracy over the
# default render resolution without being excessively slow.
_OCR_RENDER_ZOOM = 2.0


def extract_pdf(file_path: Path) -> List[ExtractedUnit]:
    """Extract text from every non-empty page of a PDF.

    Raises DocumentProcessingError for corrupt or password-protected
    files. OCR failures on individual pages are logged and skipped
    rather than aborting the whole document.
    """
    try:
        doc = fitz.open(str(file_path))
    except Exception as exc:
        raise DocumentProcessingError(f"Could not open PDF: {exc}") from exc

    if doc.is_encrypted:
        try:
            # Attempt to open with an empty password (some PDFs are
            # "encrypted" only to restrict editing, not reading).
            if not doc.authenticate(""):
                doc.close()
                raise DocumentProcessingError(
                    "PDF is password-protected and cannot be read."
                )
        except Exception as exc:
            doc.close()
            raise DocumentProcessingError(
                f"PDF is password-protected and cannot be read: {exc}"
            ) from exc

    units: List[ExtractedUnit] = []
    try:
        for page_index in range(doc.page_count):
            page = doc.load_page(page_index)
            page_number = page_index + 1
            native_text = page.get_text("text") or ""

            ocr_used = False
            text = native_text
            if len(native_text.strip()) < settings.PDF_OCR_MIN_CHARS:
                if settings.OCR_ENABLED:
                    ocr_text = _ocr_page(page, page_number)
                    if ocr_text and len(ocr_text.strip()) > len(native_text.strip()):
                        text = ocr_text
                        ocr_used = True
                # if OCR disabled or unproductive, fall back to whatever
                # native text (possibly empty) was extracted.

            normalized = normalize_text(text)
            if is_effectively_empty(normalized):
                continue

            units.append(
                ExtractedUnit(
                    text=normalized,
                    page_number=page_number,
                    ocr_used=ocr_used,
                )
            )
    finally:
        doc.close()

    return units


def _ocr_page(page: "fitz.Page", page_number: int) -> str:
    try:
        pix = page.get_pixmap(matrix=fitz.Matrix(_OCR_RENDER_ZOOM, _OCR_RENDER_ZOOM))
        image = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        return run_ocr(image)
    except OCRError as exc:
        logger.warning("OCR fallback failed on page %d: %s", page_number, exc.message)
        return ""
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Unexpected OCR error on page %d: %s", page_number, exc)
        return ""
