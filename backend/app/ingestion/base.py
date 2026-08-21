"""Shared data structures produced by every document processor.

Each processor (PDF, PPTX, image) converts its source format into a
list of `ExtractedUnit` objects — one per page / slide / image — which
gives the chunking module a uniform input regardless of source format.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class ExtractedUnit:
    """One unit of extracted content (a PDF page, a PPTX slide, or a
    whole image) prior to chunking."""

    text: str
    page_number: Optional[int] = None
    slide_number: Optional[int] = None
    ocr_used: bool = False
