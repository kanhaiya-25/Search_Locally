"""Repository classes encapsulating all SQL for documents, chunks, and
search history.

Keeping SQL in one layer means the rest of the codebase (services, API
routes) never writes raw SQL, which makes the schema easier to change
and the data-access code easier to test in isolation.
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from typing import List, Optional

from app.db.models import Chunk, Document, SearchRecord


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class DocumentRepository:
    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn

    def create(
        self,
        *,
        id: str,
        filename: str,
        original_name: str,
        file_type: str,
        file_size: int,
        checksum: str,
        processing_status: str = "pending",
    ) -> Document:
        self.conn.execute(
            """
            INSERT INTO documents
                (id, filename, original_name, file_type, file_size,
                 upload_timestamp, processing_status, processing_error, checksum)
            VALUES (?, ?, ?, ?, ?, ?, ?, NULL, ?)
            """,
            (
                id,
                filename,
                original_name,
                file_type,
                file_size,
                _now_iso(),
                processing_status,
                checksum,
            ),
        )
        return self.get_by_id(id)  # type: ignore[return-value]

    def get_by_id(self, document_id: str) -> Optional[Document]:
        row = self.conn.execute(
            "SELECT * FROM documents WHERE id = ?", (document_id,)
        ).fetchone()
        return Document.from_row(row) if row else None

    def get_by_checksum(self, checksum: str) -> Optional[Document]:
        row = self.conn.execute(
            "SELECT * FROM documents WHERE checksum = ?", (checksum,)
        ).fetchone()
        return Document.from_row(row) if row else None

    def list_all(self) -> List[Document]:
        rows = self.conn.execute(
            "SELECT * FROM documents ORDER BY upload_timestamp DESC"
        ).fetchall()
        return [Document.from_row(r) for r in rows]

    def update_status(
        self, document_id: str, status: str, error: Optional[str] = None
    ) -> None:
        self.conn.execute(
            "UPDATE documents SET processing_status = ?, processing_error = ? WHERE id = ?",
            (status, error, document_id),
        )

    def delete(self, document_id: str) -> None:
        # ON DELETE CASCADE removes associated chunks automatically.
        self.conn.execute("DELETE FROM documents WHERE id = ?", (document_id,))

    def count(self) -> int:
        row = self.conn.execute("SELECT COUNT(*) AS c FROM documents").fetchone()
        return row["c"]


class ChunkRepository:
    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn

    def bulk_create(self, chunks: List[Chunk]) -> None:
        self.conn.executemany(
            """
            INSERT INTO chunks
                (id, document_id, chunk_index, page_number, slide_number,
                 text, faiss_index, ocr_used)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    c.id,
                    c.document_id,
                    c.chunk_index,
                    c.page_number,
                    c.slide_number,
                    c.text,
                    c.faiss_index,
                    int(c.ocr_used),
                )
                for c in chunks
            ],
        )

    def get_by_document(self, document_id: str) -> List[Chunk]:
        rows = self.conn.execute(
            "SELECT * FROM chunks WHERE document_id = ? ORDER BY chunk_index",
            (document_id,),
        ).fetchall()
        return [Chunk.from_row(r) for r in rows]

    def get_by_faiss_indices(self, faiss_indices: List[int]) -> List[Chunk]:
        if not faiss_indices:
            return []
        placeholders = ",".join("?" for _ in faiss_indices)
        rows = self.conn.execute(
            f"SELECT * FROM chunks WHERE faiss_index IN ({placeholders})",
            faiss_indices,
        ).fetchall()
        return [Chunk.from_row(r) for r in rows]

    def get_all(self) -> List[Chunk]:
        rows = self.conn.execute("SELECT * FROM chunks ORDER BY faiss_index").fetchall()
        return [Chunk.from_row(r) for r in rows]

    def count_for_document(self, document_id: str) -> int:
        row = self.conn.execute(
            "SELECT COUNT(*) AS c FROM chunks WHERE document_id = ?", (document_id,)
        ).fetchone()
        return row["c"]

    def count_all(self) -> int:
        row = self.conn.execute("SELECT COUNT(*) AS c FROM chunks").fetchone()
        return row["c"]

    def max_faiss_index(self) -> int:
        row = self.conn.execute(
            "SELECT MAX(faiss_index) AS m FROM chunks"
        ).fetchone()
        return row["m"] if row["m"] is not None else -1


class SearchRepository:
    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn

    def record(self, query: str, top_k: int) -> None:
        self.conn.execute(
            "INSERT INTO searches (query, timestamp, top_k) VALUES (?, ?, ?)",
            (query, _now_iso(), top_k),
        )
        self.conn.commit()

    def recent(self, limit: int = 20) -> List[SearchRecord]:
        rows = self.conn.execute(
            "SELECT * FROM searches ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        return [SearchRecord.from_row(r) for r in rows]
