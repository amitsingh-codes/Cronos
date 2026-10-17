from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field, model_validator

from app.schemas.common import AI_SCORING_DISCLAIMER


class RubricCriterion(BaseModel):
    description: Annotated[str, Field(min_length=1, max_length=1000)]
    max_marks: Annotated[float, Field(gt=0, le=1000)]


class EvaluationItem(BaseModel):
    item_id: Annotated[str | None, Field(max_length=100)] = None
    question: Annotated[str, Field(min_length=1, max_length=3000)]
    reference_answer: Annotated[str, Field(min_length=1, max_length=8000)]
    rubric: Annotated[list[RubricCriterion] | None, Field(max_length=30)] = None
    rubric_text: Annotated[str | None, Field(max_length=4000)] = None
    max_marks: Annotated[float, Field(gt=0, le=1000)]
    student_answer: Annotated[str, Field(max_length=20_000)]  # length also capped by settings
    subject: Annotated[str | None, Field(max_length=100)] = None
    grade_level: Annotated[str | None, Field(max_length=100)] = None

    @model_validator(mode="after")
    def _rubric_consistent(self) -> "EvaluationItem":
        has_text = bool(self.rubric_text and self.rubric_text.strip())
        if not self.rubric and not has_text:
            raise ValueError("Provide a rubric (list of criteria) or rubric_text")
        if self.rubric:
            total = sum(c.max_marks for c in self.rubric)
            if abs(total - self.max_marks) > 1e-6:
                raise ValueError(f"Rubric criteria marks sum to {total}, but max_marks is {self.max_marks}")
        return self


class EvaluationBatchRequest(BaseModel):
    items: Annotated[list[EvaluationItem], Field(min_length=1, max_length=200)]  # real cap in settings


# --- Gemini output schema (flat) -----------------------------------------------
class CriterionScoreLLM(BaseModel):
    criterion_index: int  # 1-based index into the rubric list
    awarded_marks: float
    justification: str


class EvaluationLLM(BaseModel):
    criterion_scores: list[CriterionScoreLLM]
    awarded_marks: float
    correct_concepts: list[str]
    missing_key_points: list[str]
    factual_errors: list[str]
    irrelevant_content: list[str]
    explanation: str
    feedback: str
    confidence: float  # 0..1


# --- API output ----------------------------------------------------------------
class CriterionScore(BaseModel):
    criterion: str
    max_marks: float
    awarded_marks: float
    justification: str


class EvaluationResult(BaseModel):
    awarded_marks: float
    max_marks: float
    criterion_scores: list[CriterionScore]
    correct_concepts: list[str]
    missing_key_points: list[str]
    factual_errors: list[str]
    irrelevant_content: list[str]
    explanation: str
    feedback: str
    confidence: float
    needs_human_review: bool
    review_reasons: list[str]
    disclaimer: str = AI_SCORING_DISCLAIMER


class BatchItemOutcome(BaseModel):
    index: int
    item_id: str | None
    status: Literal["ok", "error"]
    result: EvaluationResult | None = None
    error: dict | None = None


class BatchSummary(BaseModel):
    total: int
    succeeded: int
    failed: int
    needs_human_review: int


class EvaluationBatchResponse(BaseModel):
    results: list[BatchItemOutcome]
    summary: BatchSummary
    disclaimer: str = AI_SCORING_DISCLAIMER
