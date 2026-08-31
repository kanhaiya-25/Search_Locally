"""Retrieval orchestration: query embedding -> FAISS search -> metadata
join -> filtering -> ranked results.

Supports three retrieval modes:
  - "semantic"  : embedding similarity only (works without any keyword index)
  - "keyword"   : BM25 lexical search only (works without the embedding model)
  - "hybrid"    : alpha * semantic_score + (1 - alpha) * keyword_score

Semantic search is the system's core capability and must function with
no LLM configured at all — this module has no dependency on the RAG
layer.
"""
from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

from app.config import settings
from app.core.exceptions import InvalidQueryError, VectorIndexError
from app.core.logging import get_logger
from app.db.repositories import ChunkRepository, DocumentRepository, SearchRepository
from app.embeddings.embedding_service import EmbeddingService
from app.retrieval.faiss_store import FaissStore
from app.retrieval.keyword_search import build_keyword_index

logger = get_logger(__name__)

MAX_QUERY_LENGTH = 1000


@dataclass
class SearchFilters:
    filename: Optional[str] = None
    file_type: Optional[str] = None
    ocr_only: Optional[bool] = None
    page_number: Optional[int] = None
    document_id: Optional[str] = None


@dataclass
class SearchResultItem:
    rank: int
    chunk_id: str
    document_id: str
    filename: str
    file_type: str
    page_number: Optional[int]
    slide_number: Optional[int]
    similarity_score: float
    text_snippet: str
    ocr_used: bool


class SearchService:
    def __init__(self, conn: sqlite3.Connection, faiss_store: FaissStore):
        self.conn = conn
        self.faiss_store = faiss_store
        self.doc_repo = DocumentRepository(conn)
        self.chunk_repo = ChunkRepository(conn)
        self.search_repo = SearchRepository(conn)

    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        filters: Optional[SearchFilters] = None,
        mode: str = "semantic",
        record_history: bool = True,
    ) -> List[SearchResultItem]:
        query = (query or "").strip()
        if not query:
            raise InvalidQueryError("Query must not be empty.")
        if len(query) > MAX_QUERY_LENGTH:
            raise InvalidQueryError(
                f"Query is too long (max {MAX_QUERY_LENGTH} characters)."
            )

        top_k = top_k or settings.DEFAULT_TOP_K
        top_k = max(1, min(top_k, settings.MAX_TOP_K))
        filters = filters or SearchFilters()

        # Over-fetch candidates before applying metadata filters, since
        # filters are applied post-retrieval (documented limitation: a
        # narrow filter combined with a small top_k may under-return).
        candidate_k = min(top_k * 5, max(top_k, self.faiss_store.ntotal) or top_k)

        start = time.perf_counter()
        scores_by_faiss_id = self._score_candidates(query, candidate_k, mode)
        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.info(
            "Search mode=%s query_len=%d candidates=%d latency_ms=%.1f",
            mode,
            len(query),
            len(scores_by_faiss_id),
            elapsed_ms,
        )

        if record_history:
            self.search_repo.record(query, top_k)

        if not scores_by_faiss_id:
            return []

        chunks = self.chunk_repo.get_by_faiss_indices(list(scores_by_faiss_id.keys()))
        chunk_by_faiss_id = {c.faiss_index: c for c in chunks if c.faiss_index is not None}

        documents = {d.id: d for d in self.doc_repo.list_all()}

        results: List[SearchResultItem] = []
        for faiss_id, score in sorted(
            scores_by_faiss_id.items(), key=lambda x: x[1], reverse=True
        ):
            chunk = chunk_by_faiss_id.get(faiss_id)
            if chunk is None:
                continue
            document = documents.get(chunk.document_id)
            if document is None:
                continue

            if not self._passes_filters(chunk, document, filters):
                continue

            results.append(
                SearchResultItem(
                    rank=0,
                    chunk_id=chunk.id,
                    document_id=document.id,
                    filename=document.original_name,
                    file_type=document.file_type,
                    page_number=chunk.page_number,
                    slide_number=chunk.slide_number,
                    similarity_score=round(score, 4),
                    text_snippet=_make_snippet(chunk.text),
                    ocr_used=chunk.ocr_used,
                )
            )
            if len(results) >= top_k:
                break

        for i, r in enumerate(results, start=1):
            r.rank = i

        return results

    # ------------------------------------------------------------------
    def _score_candidates(
        self, query: str, candidate_k: int, mode: str
    ) -> Dict[int, float]:
        if mode == "keyword":
            return self._keyword_scores(query, candidate_k)

        semantic_scores = self._semantic_scores(query, candidate_k)

        if mode == "semantic":
            return semantic_scores

        if mode == "hybrid":
            keyword_scores = self._keyword_scores(query, candidate_k)
            alpha = settings.HYBRID_ALPHA
            combined: Dict[int, float] = {}
            all_ids = set(semantic_scores) | set(keyword_scores)
            for cid in all_ids:
                combined[cid] = alpha * semantic_scores.get(cid, 0.0) + (
                    1 - alpha
                ) * keyword_scores.get(cid, 0.0)
            return combined

        raise InvalidQueryError(f"Unknown retrieval mode: {mode}")

    def _semantic_scores(self, query: str, candidate_k: int) -> Dict[int, float]:
        try:
            embedding_service = EmbeddingService.get_instance()
            query_vector = embedding_service.embed_query(query)
        except Exception as exc:
            raise VectorIndexError(f"Could not embed query: {exc}") from exc

        ids, scores = self.faiss_store.search(query_vector, candidate_k)
        return dict(zip(ids, scores))

    def _keyword_scores(self, query: str, candidate_k: int) -> Dict[int, float]:
        all_chunks = self.chunk_repo.get_all()
        indexed = [c for c in all_chunks if c.faiss_index is not None]
        if not indexed:
            return {}
        index = build_keyword_index(
            [c.faiss_index for c in indexed], [c.text for c in indexed]
        )
        return index.search(query, candidate_k)

    @staticmethod
    def _passes_filters(chunk, document, filters: SearchFilters) -> bool:
        if filters.filename and filters.filename.lower() not in document.original_name.lower():
            return False
        if filters.file_type and filters.file_type.lower() != document.file_type.lower():
            return False
        if filters.ocr_only is True and not chunk.ocr_used:
            return False
        if filters.page_number is not None and chunk.page_number != filters.page_number:
            return False
        if filters.document_id and filters.document_id != document.id:
            return False
        return True


def _make_snippet(text: str, max_chars: int = 400) -> str:
    if len(text) <= max_chars:
        return text
    truncated = text[:max_chars].rsplit(" ", 1)[0]
    return truncated + "..."
