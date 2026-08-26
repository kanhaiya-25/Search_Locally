"""Reusable text normalization pipeline.

Applied uniformly to text extracted from PDFs, PPTX slides, and OCR
output before chunking, so downstream modules always work with
consistently formatted text. Normalization is deliberately mild: it
fixes whitespace/line-break artifacts without stripping punctuation or
collapsing paragraph structure, since aggressive cleaning can destroy
information that matters for retrieval (e.g. numbered lists).
"""
from __future__ import annotations

import re

# Repeated form-feed / control characters occasionally left by PDF
# extractors.
_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
_MULTI_SPACE_RE = re.compile(r"[ \t]+")
_MULTI_BLANK_LINE_RE = re.compile(r"\n{3,}")
# A line break in the middle of a sentence (lowercase-hyphen-newline-lowercase)
# is a common PDF line-wrap artifact from justified text, e.g. "hyphen-\nated".
_HYPHEN_LINEBREAK_RE = re.compile(r"(\w)-\n(\w)")


def normalize_text(raw_text: str) -> str:
    """Normalize extracted text while preserving paragraph structure.

    Steps:
      1. Strip control characters.
      2. Rejoin words that were hyphen-split across a line wrap.
      3. Collapse runs of spaces/tabs to a single space.
      4. Collapse 3+ consecutive blank lines to a single blank line.
      5. Strip leading/trailing whitespace on each line and overall.
    """
    if not raw_text:
        return ""

    text = _CONTROL_CHARS_RE.sub("", raw_text)
    text = _HYPHEN_LINEBREAK_RE.sub(r"\1\2", text)
    text = _MULTI_SPACE_RE.sub(" ", text)

    lines = [line.strip() for line in text.split("\n")]
    text = "\n".join(lines)
    text = _MULTI_BLANK_LINE_RE.sub("\n\n", text)

    return text.strip()


def is_effectively_empty(text: str, min_chars: int = 3) -> bool:
    """Return True if text has fewer than `min_chars` non-whitespace
    characters (used to skip blank pages/slides)."""
    return len(re.sub(r"\s+", "", text or "")) < min_chars
