from __future__ import annotations

from pydantic import BaseModel


class RagSource(BaseModel):
    chunk_id: int
    page: int | None
    score: float
    excerpt: str


class RagAnswerLLM(BaseModel):
    """Schema Gemini must follow (kept flat: no defaults/constraints)."""

    answerable: bool
    answer: str


class RagResponse(BaseModel):
    answer: str
    answerable: bool
    sources: list[RagSource]
    metadata: dict
