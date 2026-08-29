"""FAISS vector index management.

Similarity method: cosine similarity via normalized inner product.
Embeddings produced by EmbeddingService are L2-normalized, and the
index type used here is `IndexIDMap2(IndexFlatIP(dim))`:

  - `IndexFlatIP` computes an exact inner product between the query and
    every stored vector. Because all vectors are unit-normalized, the
    inner product is mathematically equivalent to cosine similarity,
    bounded in [-1, 1] (in practice [0, 1] for sentence embeddings of
    natural text).
  - `IndexIDMap2` wraps the flat index so we can assign our own stable
    integer ids (matching `chunks.faiss_index` in SQLite) instead of
    relying on FAISS's implicit insertion order, and so individual
    vectors can be removed by id without a full rebuild.

The index is persisted to disk after every mutation and reloaded at
application startup. A full rebuild (used by the /reindex endpoint) is
written to a temporary file and only swapped in after it completes
successfully, so a failed rebuild never corrupts the active index.
"""
from __future__ import annotations

import os
import tempfile
from typing import List, Tuple

import faiss
import numpy as np

from app.config import settings
from app.core.exceptions import VectorIndexError
from app.core.logging import get_logger

logger = get_logger(__name__)


class FaissStore:
    def __init__(self, dimension: int):
        self.dimension = dimension
        self.index_path = settings.FAISS_INDEX_PATH
        self._index = self._load_or_create()

    # ------------------------------------------------------------------
    # Construction / persistence
    # ------------------------------------------------------------------
    def _new_index(self) -> faiss.Index:
        flat = faiss.IndexFlatIP(self.dimension)
        return faiss.IndexIDMap2(flat)

    def _load_or_create(self) -> faiss.Index:
        if self.index_path.exists():
            try:
                index = faiss.read_index(str(self.index_path))
                if index.d != self.dimension:
                    logger.warning(
                        "Existing FAISS index dimension (%d) does not match "
                        "embedding model dimension (%d); creating a fresh index.",
                        index.d,
                        self.dimension,
                    )
                    return self._new_index()
                logger.info(
                    "Loaded FAISS index from %s (%d vectors)",
                    self.index_path,
                    index.ntotal,
                )
                return index
            except Exception as exc:
                logger.error("Failed to load FAISS index, creating a new one: %s", exc)
                return self._new_index()
        logger.info("No existing FAISS index found; creating a new one.")
        return self._new_index()

    def save(self) -> None:
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        # Write to a temp file then atomically replace, so a crash
        # mid-write never leaves a corrupt index on disk.
        fd, tmp_path = tempfile.mkstemp(
            dir=str(self.index_path.parent), suffix=".index.tmp"
        )
        os.close(fd)
        try:
            faiss.write_index(self._index, tmp_path)
            os.replace(tmp_path, str(self.index_path))
        except Exception as exc:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise VectorIndexError(f"Failed to save FAISS index: {exc}") from exc

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------
    def add(self, ids: List[int], vectors: np.ndarray) -> None:
        if len(ids) == 0:
            return
        if vectors.shape[0] != len(ids):
            raise VectorIndexError("Vector count does not match id count.")
        id_array = np.array(ids, dtype=np.int64)
        self._index.add_with_ids(vectors.astype(np.float32), id_array)
        self.save()

    def remove(self, ids: List[int]) -> None:
        if not ids:
            return
        id_array = np.array(ids, dtype=np.int64)
        selector = faiss.IDSelectorArray(id_array)
        self._index.remove_ids(selector)
        self.save()

    def rebuild(self, ids: List[int], vectors: np.ndarray) -> None:
        """Atomically replace the index contents with the given ids and
        vectors. Used by the /reindex endpoint and on startup consistency
        repair."""
        new_index = self._new_index()
        if len(ids) > 0:
            id_array = np.array(ids, dtype=np.int64)
            new_index.add_with_ids(vectors.astype(np.float32), id_array)
        self._index = new_index
        self.save()

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------
    def search(self, query_vector: np.ndarray, top_k: int) -> Tuple[List[int], List[float]]:
        if self._index.ntotal == 0:
            return [], []
        query = query_vector.reshape(1, -1).astype(np.float32)
        scores, ids = self._index.search(query, min(top_k, self._index.ntotal))
        result_ids = [int(i) for i in ids[0] if i != -1]
        result_scores = [float(s) for i, s in zip(ids[0], scores[0]) if i != -1]
        return result_ids, result_scores

    @property
    def ntotal(self) -> int:
        return self._index.ntotal
