"""Local embedding generation using Sentence-Transformers.

The model is loaded once (lazily, on first use) and reused for every
subsequent call — reloading a transformer model per request would be
prohibitively slow. Embeddings are L2-normalized so that FAISS's inner
product index (IndexFlatIP) computes cosine similarity; see
retrieval/faiss_store.py for the consuming side of this contract.
"""
from __future__ import annotations

from typing import List

import numpy as np

from app.config import settings
from app.core.exceptions import EmbeddingServiceError
from app.core.logging import get_logger

logger = get_logger(__name__)


class EmbeddingService:
    """Singleton-style wrapper around a SentenceTransformer model."""

    _instance: "EmbeddingService | None" = None

    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or settings.EMBEDDING_MODEL
        self._model = None

    @classmethod
    def get_instance(cls) -> "EmbeddingService":
        if cls._instance is None:
            cls._instance = EmbeddingService()
        return cls._instance

    def _load_model(self):
        if self._model is None:
            try:
                # Imported lazily so the (relatively heavy) torch/ST
                # import only happens when embeddings are actually needed.
                from sentence_transformers import SentenceTransformer

                logger.info("Loading embedding model '%s'...", self.model_name)
                self._model = SentenceTransformer(self.model_name)
                logger.info("Embedding model loaded.")
            except Exception as exc:
                raise EmbeddingServiceError(
                    f"Failed to load embedding model '{self.model_name}': {exc}"
                ) from exc
        return self._model

    @property
    def dimension(self) -> int:
        model = self._load_model()
        # sentence-transformers >= 5 renamed this method; support both.
        if hasattr(model, "get_embedding_dimension"):
            return model.get_embedding_dimension()
        return model.get_sentence_embedding_dimension()

    def is_available(self) -> bool:
        try:
            self._load_model()
            return True
        except EmbeddingServiceError:
            return False

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Embed a list of texts in batches, returning an (N, D) float32
        array of L2-normalized vectors."""
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)

        model = self._load_model()
        try:
            embeddings = model.encode(
                texts,
                batch_size=settings.EMBEDDING_BATCH_SIZE,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
        except Exception as exc:
            raise EmbeddingServiceError(f"Embedding generation failed: {exc}") from exc

        return embeddings.astype(np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query string, returning a (D,) float32 vector."""
        return self.embed_texts([query])[0]
