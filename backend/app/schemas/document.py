"""Pydantic request/response models for the document endpoints."""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class DocumentResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    original_name: str
    file_type: str
    file_size: int
    upload_timestamp: str
    processing_status: str
    processing_error: Optional[str] = None
    chunk_count: int = 0


class UploadResponse(BaseModel):
    success: bool
    document: Optional[DocumentResponse] = None
    message: str
    duplicate_of: Optional[str] = None


class DeleteResponse(BaseModel):
    success: bool
    message: str


class ReindexResponse(BaseModel):
    success: bool
    chunks_reindexed: int
    message: str


class ErrorResponse(BaseModel):
    detail: str = Field(..., description="Human-readable error message")
