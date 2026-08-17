"""SQLite connection management and schema creation.

Raw `sqlite3` is used instead of a full ORM. For a project of this size
a thin wrapper keeps the schema, the SQL, and the data flow transparent
and easy to explain in a viva, while still giving us foreign keys,
indexes, and parameterized queries.
"""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from typing import Iterator

from app.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id                  TEXT PRIMARY KEY,
    filename            TEXT NOT NULL,
    original_name       TEXT NOT NULL,
    file_type           TEXT NOT NULL,
    file_size           INTEGER NOT NULL,
    upload_timestamp    TEXT NOT NULL,
    processing_status   TEXT NOT NULL DEFAULT 'pending',
    processing_error    TEXT,
    checksum            TEXT NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS chunks (
    id              TEXT PRIMARY KEY,
    document_id     TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index     INTEGER NOT NULL,
    page_number     INTEGER,
    slide_number    INTEGER,
    text            TEXT NOT NULL,
    faiss_index     INTEGER UNIQUE,
    ocr_used        INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON chunks(document_id);
CREATE INDEX IF NOT EXISTS idx_chunks_faiss_index ON chunks(faiss_index);

CREATE TABLE IF NOT EXISTS searches (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    query       TEXT NOT NULL,
    timestamp   TEXT NOT NULL,
    top_k       INTEGER NOT NULL
);
"""


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(str(settings.DATABASE_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db() -> None:
    """Create tables if they do not already exist. Safe to call on every
    application startup."""
    settings.ensure_directories()
    conn = get_connection()
    try:
        conn.executescript(SCHEMA)
        conn.commit()
        logger.info("Database initialized at %s", settings.DATABASE_PATH)
    finally:
        conn.close()


@contextmanager
def db_session() -> Iterator[sqlite3.Connection]:
    """Context manager yielding a connection with automatic commit/rollback."""
    conn = get_connection()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
