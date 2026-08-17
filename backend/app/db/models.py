"""Plain dataclasses representing rows returned from SQLite.

These are intentionally simple data containers (not an ORM's mapped
classes) — they exist so the rest of the codebase can use typed
attribute access (`doc.filename`) instead of dict/Row indexing
everywhere.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from typing import Optional


@dataclass
class Document:
    id: str
    filename: str
    original_name: str
    file_type: str
    file_size: int
    upload_timestamp: str
    processing_status: str
    processing_error: Optional[str]
    checksum: str

    @staticmethod
    def from_row(row: sqlite3.Row) -> "Document":
        return Document(
            id=row["id"],
            filename=row["filename"],
            original_name=row["original_name"],
            file_type=row["file_type"],
            file_size=row["file_size"],
            upload_timestamp=row["upload_timestamp"],
            processing_status=row["processing_status"],
            processing_error=row["processing_error"],
            checksum=row["checksum"],
        )


@dataclass
class Chunk:
    id: str
    document_id: str
    chunk_index: int
    page_number: Optional[int]
    slide_number: Optional[int]
    text: str
    faiss_index: Optional[int]
    ocr_used: bool

    @staticmethod
    def from_row(row: sqlite3.Row) -> "Chunk":
        return Chunk(
            id=row["id"],
            document_id=row["document_id"],
            chunk_index=row["chunk_index"],
            page_number=row["page_number"],
            slide_number=row["slide_number"],
            text=row["text"],
            faiss_index=row["faiss_index"],
            ocr_used=bool(row["ocr_used"]),
        )


@dataclass
class SearchRecord:
    id: int
    query: str
    timestamp: str
    top_k: int

    @staticmethod
    def from_row(row: sqlite3.Row) -> "SearchRecord":
        return SearchRecord(
            id=row["id"],
            query=row["query"],
            timestamp=row["timestamp"],
            top_k=row["top_k"],
        )
