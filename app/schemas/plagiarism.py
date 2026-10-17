from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

PLAGIARISM_DISCLAIMER = (
    "Similarity is a screening signal, not proof of plagiarism or copying. Shared source material, "
    "quotations, common phrasing and templates produce overlap. A person must review flagged pairs; "
    "no disciplinary decision should be made from this output alone."
)
PARAPHRASE_LIMITATIONS = (
    "Gemini's paraphrase assessment is a probabilistic opinion on a few sentence pairs only, "
    "can be wrong, and must not be treated as evidence by itself."
)


class DocumentInfo(BaseModel):
    name: str
    characters: int
    meaningful_words: int
    used_ocr: bool
    status: Literal["compared", "insufficient_text"]


class SkippedDocument(BaseModel):
    name: str
    reason: str


class MatchingSentencePair(BaseModel):
    similarity: float
    excerpt_a: str
    excerpt_b: str


class SentenceOverlap(BaseModel):
    sentences_compared_a: int
    sentences_compared_b: int
    matched_a: int
    matched_b: int
    matched_ratio_a: float
    matched_ratio_b: float
    exact_matches: int
    top_matches: list[MatchingSentencePair] | None = None  # only if include_excerpts


class ParaphraseAssessment(BaseModel):
    status: Literal["completed", "skipped", "unavailable"]
    likely_paraphrase: bool | None = None
    confidence: float | None = None
    explanation: str | None = None
    note: str
    limitations: str = PARAPHRASE_LIMITATIONS


class ParaphraseLLM(BaseModel):
    likely_paraphrase: bool
    confidence: float
    explanation: str


class PairResult(BaseModel):
    document_a: str
    document_b: str
    document_similarity: float  # 0..1 TF-IDF cosine
    document_similarity_percent: float
    sentence_overlap: SentenceOverlap
    flagged_for_review: bool
    flag_reasons: list[str]
    paraphrase_assessment: ParaphraseAssessment | None = None


class PlagiarismResponse(BaseModel):
    documents: list[DocumentInfo]
    skipped_documents: list[SkippedDocument]
    pairs: list[PairResult]
    thresholds: dict
    warnings: list[str]
    disclaimer: str = PLAGIARISM_DISCLAIMER
