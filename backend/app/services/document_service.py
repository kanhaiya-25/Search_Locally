"""Document ingestion, deletion, and reindexing orchestration.

This is the glue module that ties together upload validation,
duplicate detection, format-specific extraction, chunking, embedding,
FAISS indexing, and SQLite persistence. Each stage is implemented in
its own module (ingestion/, chunking/, embeddings/, retrieval/); this
service only sequences them and handles cross-cutting concerns
(status tracking, error isolation per document, id/index consistency).
"""
from __future__ import annotations

import sqlite3
import uuid
from pathlib import Path
from typing import List, Optional

from app.chunking.chunker import TextChunk, chunk_units
from app.config import settings
from app.core.exceptions import (
    DocumentNotFoundError,
    DocumentProcessingError,
    DuplicateDocumentError,
    EmptyFileError,
    FileTooLargeError,
    UnsupportedFileTypeError,
)
from app.core.logging import get_logger
from app.db.models import Chunk, Document
from app.db.repositories import ChunkRepository, DocumentRepository
from app.embeddings.embedding_service import EmbeddingService
from app.ingestion.base import ExtractedUnit
from app.ingestion.image_processor import extract_image
from app.ingestion.pdf_processor import extract_pdf
from app.ingestion.ppt_processor import extract_pptx
from app.retrieval.faiss_store import FaissStore
from app.utils.checksum import compute_sha256
from app.utils.security import build_storage_path, sanitize_display_name, validate_extension

logger = get_logger(__name__)

_EXTRACTORS = {
    ".pdf": extract_pdf,
    ".pptx": extract_pptx,
    ".ppt": extract_pptx,  # raises a clear DocumentProcessingError internally
    ".png": extract_image,
    ".jpg": extract_image,
    ".jpeg": extract_image,
}

_FILE_TYPE_LABELS = {
    ".pdf": "pdf",
    ".pptx": "pptx",
    ".ppt": "ppt",
    ".png": "image",
    ".jpg": "image",
    ".jpeg": "image",
}


class DocumentService:
    def __init__(
        self,
        conn: sqlite3.Connection,
        faiss_store: FaissStore,
        embedding_service: Optional[EmbeddingService] = None,
    ):
        self.conn = conn
        self.faiss_store = faiss_store
        self.embedding_service = embedding_service or EmbeddingService.get_instance()
        self.doc_repo = DocumentRepository(conn)
        self.chunk_repo = ChunkRepository(conn)

    # ------------------------------------------------------------------
    # Upload
    # ------------------------------------------------------------------
    def upload_document(self, original_filename: str, file_bytes: bytes) -> Document:
        if len(file_bytes) == 0:
            raise EmptyFileError("Uploaded file is empty.")

        max_bytes = settings.MAX_FILE_SIZE_MB * 1024 * 1024
        if len(file_bytes) > max_bytes:
            raise FileTooLargeError(
                f"File exceeds the maximum allowed size of {settings.MAX_FILE_SIZE_MB} MB."
            )

        extension = validate_extension(original_filename)
        display_name = sanitize_display_name(original_filename)
        checksum = compute_sha256(file_bytes)

        existing = self.doc_repo.get_by_checksum(checksum)
        if existing is not None:
            raise DuplicateDocumentError(
                f"This file has already been uploaded as '{existing.original_name}'.",
                existing_document_id=existing.id,
            )

        document_id = str(uuid.uuid4())
        storage_path = build_storage_path(document_id, extension)
        storage_path.write_bytes(file_bytes)

        document = self.doc_repo.create(
            id=document_id,
            filename=storage_path.name,
            original_name=display_name,
            file_type=_FILE_TYPE_LABELS[extension],
            file_size=len(file_bytes),
            checksum=checksum,
            processing_status="pending",
        )
        self.conn.commit()

        self._process_document(document, storage_path, extension)
        return self.doc_repo.get_by_id(document_id)  # type: ignore[return-value]

    def _process_document(self, document: Document, storage_path: Path, extension: str) -> None:
        """Extract, chunk, embed, and index a single document. Failures
        here are caught and stored as `processing_error` on the document
        row rather than propagated, so one bad file never aborts a batch
        upload."""
        try:
            self.doc_repo.update_status(document.id, "processing")
            self.conn.commit()

            extractor = _EXTRACTORS.get(extension)
            if extractor is None:
                raise UnsupportedFileTypeError(f"No extractor registered for '{extension}'.")

            units: List[ExtractedUnit] = extractor(storage_path)
            if not units:
                self.doc_repo.update_status(
                    document.id,
                    "completed_no_text",
                    "No extractable text was found in this document.",
                )
                self.conn.commit()
                logger.warning("Document %s produced no extractable text.", document.id)
                return

            text_chunks: List[TextChunk] = chunk_units(document.id, units)
            if not text_chunks:
                self.doc_repo.update_status(
                    document.id, "completed_no_text", "No usable chunks were produced."
                )
                self.conn.commit()
                return

            embeddings = self.embedding_service.embed_texts([c.text for c in text_chunks])

            start_faiss_id = self.chunk_repo.max_faiss_index() + 1
            faiss_ids = list(range(start_faiss_id, start_faiss_id + len(text_chunks)))

            db_chunks = [
                Chunk(
                    id=tc.id,
                    document_id=tc.document_id,
                    chunk_index=tc.chunk_index,
                    page_number=tc.page_number,
                    slide_number=tc.slide_number,
                    text=tc.text,
                    faiss_index=faiss_id,
                    ocr_used=tc.ocr_used,
                )
                for tc, faiss_id in zip(text_chunks, faiss_ids)
            ]

            self.chunk_repo.bulk_create(db_chunks)
            self.faiss_store.add(faiss_ids, embeddings)

            self.doc_repo.update_status(document.id, "completed", None)
            self.conn.commit()
            logger.info(
                "Document %s processed successfully: %d chunks indexed.",
                document.id,
                len(db_chunks),
            )

        except DocumentProcessingError as exc:
            self.conn.rollback()
            self.doc_repo.update_status(document.id, "failed", exc.message)
            self.conn.commit()
            logger.error("Processing failed for document %s: %s", document.id, exc.message)
        except Exception as exc:  # pragma: no cover - defensive catch-all
            self.conn.rollback()
            self.doc_repo.update_status(document.id, "failed", str(exc))
            self.conn.commit()
            logger.exception("Unexpected error processing document %s", document.id)

    # ------------------------------------------------------------------
    # Deletion
    # ------------------------------------------------------------------
    def delete_document(self, document_id: str) -> None:
        document = self.doc_repo.get_by_id(document_id)
        if document is None:
            raise DocumentNotFoundError(f"Document '{document_id}' not found.")

        chunks = self.chunk_repo.get_by_document(document_id)
        faiss_ids = [c.faiss_index for c in chunks if c.faiss_index is not None]

        self.faiss_store.remove(faiss_ids)  # safe no-op if empty

        self.doc_repo.delete(document_id)  # cascades to chunks
        self.conn.commit()

        storage_path = settings.UPLOAD_DIR / document.filename
        if storage_path.exists():
            storage_path.unlink()

        logger.info("Deleted document %s and %d chunks.", document_id, len(faiss_ids))

    # ------------------------------------------------------------------
    # Reindexing
    # ------------------------------------------------------------------
    def reindex_all(self) -> int:
        """Regenerate embeddings for every stored chunk and rebuild the
        FAISS index from scratch, atomically. Returns the number of
        chunks reindexed. If regeneration fails, the previously active
        index on disk is left untouched (see FaissStore.rebuild)."""
        all_chunks = self.chunk_repo.get_all()
        if not all_chunks:
            self.faiss_store.rebuild([], self.embedding_service.embed_texts([]))
            return 0

        texts = [c.text for c in all_chunks]
        embeddings = self.embedding_service.embed_texts(texts)
        faiss_ids = [c.faiss_index for c in all_chunks]

        self.faiss_store.rebuild(faiss_ids, embeddings)
        logger.info("Reindex complete: %d chunks.", len(all_chunks))
        return len(all_chunks)
