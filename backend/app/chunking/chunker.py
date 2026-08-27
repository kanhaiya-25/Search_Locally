"""Configurable word-based chunking with overlap.

Each ExtractedUnit (a page or slide) is split into overlapping,
word-count-bounded chunks. Word count is used instead of a tokenizer
for simplicity and transparency (no extra tokenizer dependency); it is
a reasonable proxy for token count for English text (~0.75 tokens per
word on average for common transformer tokenizers).

The chunker never merges text across two different source pages/slides,
so every chunk keeps a single, unambiguous page/slide reference.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import List

from app.config import settings
from app.ingestion.base import ExtractedUnit


@dataclass
class TextChunk:
    id: str
    document_id: str
    chunk_index: int
    page_number: int | None
    slide_number: int | None
    text: str
    ocr_used: bool


def chunk_units(document_id: str, units: List[ExtractedUnit]) -> List[TextChunk]:
    """Chunk every extracted unit belonging to a document, producing a
    single, globally chunk-indexed list for that document."""
    chunk_size = settings.CHUNK_SIZE_WORDS
    overlap = settings.CHUNK_OVERLAP_WORDS
    min_size = settings.MIN_CHUNK_SIZE_WORDS
    max_size = settings.MAX_CHUNK_SIZE_WORDS

    if overlap >= chunk_size:
        overlap = max(0, chunk_size // 5)  # defensive: keep step positive

    chunks: List[TextChunk] = []
    chunk_index = 0

    for unit in units:
        words = unit.text.split()
        if not words:
            continue

        if len(words) <= max_size:
            word_groups = _split_with_overlap(words, chunk_size, overlap)
        else:
            # Very long unit (rare: a dense single PDF page) — still
            # split with the same sliding window so no chunk exceeds
            # max_size by more than the configured chunk size.
            word_groups = _split_with_overlap(words, chunk_size, overlap)

        for group in word_groups:
            if len(group) < min_size and len(word_groups) > 1:
                # Merge tiny trailing fragments into the previous chunk
                # instead of creating a near-empty chunk.
                if chunks and chunks[-1].document_id == document_id:
                    chunks[-1].text = chunks[-1].text + " " + " ".join(group)
                    continue

            chunks.append(
                TextChunk(
                    id=str(uuid.uuid4()),
                    document_id=document_id,
                    chunk_index=chunk_index,
                    page_number=unit.page_number,
                    slide_number=unit.slide_number,
                    text=" ".join(group),
                    ocr_used=unit.ocr_used,
                )
            )
            chunk_index += 1

    return chunks


def _split_with_overlap(
    words: List[str], chunk_size: int, overlap: int
) -> List[List[str]]:
    step = max(1, chunk_size - overlap)
    groups: List[List[str]] = []
    start = 0
    n = len(words)
    while start < n:
        end = min(start + chunk_size, n)
        groups.append(words[start:end])
        if end == n:
            break
        start += step
    return groups
