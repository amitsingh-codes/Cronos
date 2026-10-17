from __future__ import annotations

from enum import Enum
from typing import Annotated

from pydantic import BaseModel, Field, field_validator, model_validator


class Difficulty(str, Enum):
    easy = "easy"
    medium = "medium"
    hard = "hard"


class QuestionType(str, Enum):
    mcq = "mcq"
    short_answer = "short_answer"
    long_answer = "long_answer"
    fill_blank = "fill_blank"


class QuestionRequest(BaseModel):
    source_text: Annotated[str, Field(min_length=50, max_length=30_000)]
    num_questions: Annotated[int, Field(ge=1, le=50)]
    difficulty: Difficulty
    question_types: Annotated[list[QuestionType], Field(min_length=1)] = [QuestionType.mcq]
    total_marks: Annotated[int | None, Field(ge=1, le=1000)] = None
    subject: Annotated[str | None, Field(max_length=100)] = None
    instructions: Annotated[str | None, Field(max_length=1000)] = None

    @field_validator("source_text")
    @classmethod
    def _not_blank(cls, v: str) -> str:
        if len(v.strip()) < 50:
            raise ValueError("source_text must contain at least 50 non-whitespace characters")
        return v

    @field_validator("question_types")
    @classmethod
    def _unique_types(cls, v: list[QuestionType]) -> list[QuestionType]:
        if len(set(v)) != len(v):
            raise ValueError("question_types must not contain duplicates")
        return v

    @model_validator(mode="after")
    def _marks_feasible(self) -> "QuestionRequest":
        if self.total_marks is not None and self.total_marks < self.num_questions:
            raise ValueError("total_marks must be at least num_questions (minimum 1 mark per question)")
        return self


# --- what Gemini must return (flat, no constraints) --------------------------
class GeneratedQuestionLLM(BaseModel):
    text: str
    options: list[str] | None
    correct_option: str | None  # "A".."D" for MCQ, otherwise null
    correct_answer: str
    explanation: str
    source_excerpt: str  # short verbatim quote from the source text supporting the answer


class GeneratedQuestionsLLM(BaseModel):
    questions: list[GeneratedQuestionLLM]


# --- what the API returns ------------------------------------------------------
class Question(BaseModel):
    number: int
    type: QuestionType
    text: str
    options: list[str] | None = None
    correct_option: str | None = None
    correct_answer: str
    marks: int
    explanation: str
    source_excerpt: str
    evidence_verified: bool  # True if source_excerpt was found verbatim in the source text


class QuestionPaperResponse(BaseModel):
    questions: list[Question]
    total_marks: int
    difficulty: Difficulty
    subject: str | None
    warnings: list[str]
    disclaimer: str
