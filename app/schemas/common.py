from __future__ import annotations

from pydantic import BaseModel


class ErrorDetail(BaseModel):
    code: str
    message: str
    details: list | None = None


class ErrorResponse(BaseModel):
    error: ErrorDetail


class HealthResponse(BaseModel):
    status: str
    version: str
    gemini_configured: bool
    ocr_available: bool


AI_SCORING_DISCLAIMER = (
    "AI-assisted output. It can be wrong and is not a substitute for teacher judgement; "
    "review flagged items and spot-check the rest before using it for grading."
)
