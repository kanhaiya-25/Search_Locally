"""Pydantic request/response models for the RAG question-answering
endpoint."""
from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field

from app.schemas.search import SearchFiltersSchema, SearchResultSchema


class QARequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000)
    top_k: int = Field(default=5, ge=1, le=15)
    filters: Optional[SearchFiltersSchema] = None


class CitationSchema(BaseModel):
    filename: str
    page_number: Optional[int] = None
    slide_number: Optional[int] = None


class QAResponse(BaseModel):
    question: str
    answer: Optional[str] = None
    llm_used: bool
    sufficient_context: bool
    citations: List[CitationSchema]
    supporting_passages: List[SearchResultSchema]
    message: str
