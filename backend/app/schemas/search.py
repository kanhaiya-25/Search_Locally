"""Pydantic request/response models for the search endpoints."""
from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class SearchFiltersSchema(BaseModel):
    filename: Optional[str] = None
    file_type: Optional[str] = None
    ocr_only: Optional[bool] = None
    page_number: Optional[int] = None
    document_id: Optional[str] = None


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000)
    top_k: int = Field(default=5, ge=1, le=25)
    filters: Optional[SearchFiltersSchema] = None
    mode: Literal["semantic", "keyword", "hybrid"] = "semantic"


class SearchResultSchema(BaseModel):
    rank: int
    chunk_id: str
    document_id: str
    filename: str
    file_type: str
    page_number: Optional[int] = None
    slide_number: Optional[int] = None
    similarity_score: float
    text_snippet: str
    ocr_used: bool


class SearchResponse(BaseModel):
    query: str
    mode: str
    result_count: int
    results: List[SearchResultSchema]
    latency_ms: float


class SearchHistoryItem(BaseModel):
    id: int
    query: str
    timestamp: str
    top_k: int


class SearchHistoryResponse(BaseModel):
    history: List[SearchHistoryItem]
