"""
Central configuration for the backend application.

All tunable parameters are read from environment variables (with sensible
defaults) so behaviour can be changed without touching source code. See
`.env.example` at the repository root of `backend/` for the full list of
supported variables.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Load a .env file if present. This is a no-op in environments where the
# variables are already set (e.g. CI, Docker).
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent  # backend/


def _get_bool(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "on"}


def _get_int(name: str, default: int) -> int:
    val = os.getenv(name)
    if val is None or val == "":
        return default
    try:
        return int(val)
    except ValueError:
        return default


def _get_float(name: str, default: float) -> float:
    val = os.getenv(name)
    if val is None or val == "":
        return default
    try:
        return float(val)
    except ValueError:
        return default


class Settings:
    """Application settings, populated once at import time."""

    # --- Storage paths -----------------------------------------------
    DATA_DIR: Path = Path(os.getenv("DATA_DIR", str(BASE_DIR / "data")))
    UPLOAD_DIR: Path = Path(os.getenv("UPLOAD_DIR", str(DATA_DIR / "uploads")))
    FAISS_INDEX_PATH: Path = Path(
        os.getenv("FAISS_INDEX_PATH", str(DATA_DIR / "index" / "faiss.index"))
    )
    DATABASE_PATH: Path = Path(
        os.getenv("DATABASE_PATH", str(DATA_DIR / "sqlite" / "app.db"))
    )

    # --- Upload validation ---------------------------------------------
    MAX_FILE_SIZE_MB: int = _get_int("MAX_FILE_SIZE_MB", 50)
    ALLOWED_EXTENSIONS = {".pdf", ".ppt", ".pptx", ".png", ".jpg", ".jpeg"}

    # --- Text extraction / OCR -----------------------------------------
    OCR_ENABLED: bool = _get_bool("OCR_ENABLED", True)
    OCR_LANGUAGE: str = os.getenv("OCR_LANGUAGE", "eng")
    # If native text extracted from a PDF page has fewer characters than
    # this threshold, the page is treated as "scanned" and OCR fallback
    # is attempted (only if OCR_ENABLED is true).
    PDF_OCR_MIN_CHARS: int = _get_int("PDF_OCR_MIN_CHARS", 20)
    TESSERACT_CMD: str = os.getenv("TESSERACT_CMD", "")  # optional explicit path

    # --- Chunking --------------------------------------------------------
    CHUNK_SIZE_WORDS: int = _get_int("CHUNK_SIZE_WORDS", 650)
    CHUNK_OVERLAP_WORDS: int = _get_int("CHUNK_OVERLAP_WORDS", 100)
    MIN_CHUNK_SIZE_WORDS: int = _get_int("MIN_CHUNK_SIZE_WORDS", 40)
    MAX_CHUNK_SIZE_WORDS: int = _get_int("MAX_CHUNK_SIZE_WORDS", 900)

    # --- Embeddings --------------------------------------------------------
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    EMBEDDING_BATCH_SIZE: int = _get_int("EMBEDDING_BATCH_SIZE", 32)
    # Dimensionality is derived from the model at runtime, but we keep a
    # documented default for all-MiniLM-L6-v2 for reference: 384.

    # --- Retrieval --------------------------------------------------------
    DEFAULT_TOP_K: int = _get_int("TOP_K", 5)
    MAX_TOP_K: int = _get_int("MAX_TOP_K", 25)
    # Cosine similarity threshold below which results are considered too
    # weak to be trusted. Vectors are L2-normalized and FAISS uses inner
    # product, so the score returned is a cosine similarity in [-1, 1].
    SIMILARITY_THRESHOLD: float = _get_float("SIMILARITY_THRESHOLD", 0.35)
    HYBRID_ALPHA: float = _get_float("HYBRID_ALPHA", 0.6)  # weight on semantic score

    # --- RAG / LLM --------------------------------------------------------
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "none")  # none | openai
    LLM_API_KEY: str = os.getenv("LLM_API_KEY", "")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "gpt-4o-mini")
    QA_MAX_CONTEXT_CHUNKS: int = _get_int("QA_MAX_CONTEXT_CHUNKS", 5)

    # --- Misc --------------------------------------------------------
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    CORS_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",")

    def ensure_directories(self) -> None:
        """Create all directories required by the application if missing."""
        self.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        self.FAISS_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
        self.DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)


settings = Settings()
settings.ensure_directories()
