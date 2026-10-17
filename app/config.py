"""Environment-based configuration. Every tunable value lives here."""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "ScoreMatrix API"
    app_version: str = "0.1.0"
    log_level: str = "INFO"

    # --- Gemini -------------------------------------------------------
    gemini_api_key: SecretStr | None = None
    gemini_model: str = "gemini-2.5-flash"
    gemini_timeout_seconds: float = Field(60.0, gt=0)
    gemini_max_retries: int = Field(2, ge=0, le=5)  # retries AFTER the first attempt
    gemini_backoff_base_seconds: float = Field(1.0, ge=0)
    gemini_backoff_max_seconds: float = Field(8.0, ge=0)
    gemini_invalid_output_retries: int = Field(1, ge=0, le=3)

    # --- HTTP / security ----------------------------------------------
    cors_allowed_origins: str = ""  # comma-separated; empty = CORS disabled
    max_upload_mb: int = Field(10, ge=1)
    request_timeout_seconds: float = Field(120.0, gt=0)
    rate_limit_per_minute: int = Field(60, ge=0)  # 0 disables; in-memory, single process only

    # --- PDF / OCR ----------------------------------------------------
    pdf_max_pages: int = Field(300, ge=1)
    min_pdf_text_chars: int = Field(30, ge=0)  # below this a PDF is treated as image-only
    ocr_enabled: bool = True
    ocr_language: str = "eng"
    ocr_max_pages: int = Field(15, ge=1)
    ocr_dpi: int = Field(200, ge=72, le=400)
    ocr_timeout_seconds: int = Field(30, ge=1)
    ocr_max_image_pixels: int = 50_000_000

    # --- RAG ----------------------------------------------------------
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    preload_embedding_model: bool = False
    rag_chunk_size: int = Field(800, ge=100)
    rag_chunk_overlap: int = Field(100, ge=0)
    rag_top_k: int = Field(4, ge=1, le=10)
    rag_max_chunks: int = Field(2000, ge=1)
    rag_max_question_chars: int = 1000

    # --- Question generation -------------------------------------------
    max_source_chars: int = 30_000
    max_questions: int = Field(50, ge=1)

    # --- Evaluation ----------------------------------------------------
    eval_max_batch_size: int = Field(25, ge=1)
    eval_batch_concurrency: int = Field(4, ge=1)
    eval_confidence_threshold: float = Field(0.6, ge=0, le=1)
    eval_max_answer_chars: int = 10_000

    # --- Plagiarism ----------------------------------------------------
    plagiarism_max_files: int = Field(10, ge=2)
    plagiarism_min_tokens: int = Field(20, ge=1)  # fewer meaningful words => "insufficient text"
    plagiarism_doc_threshold: float = Field(0.75, ge=0, le=1)
    plagiarism_sentence_threshold: float = Field(0.8, ge=0, le=1)
    plagiarism_min_sentence_tokens: int = Field(6, ge=1)
    plagiarism_min_matching_sentences: int = Field(3, ge=1)
    plagiarism_sentence_ratio_threshold: float = Field(0.3, ge=0, le=1)
    plagiarism_max_sentences_per_doc: int = Field(500, ge=10)
    plagiarism_ai_max_pairs: int = Field(3, ge=1)
    plagiarism_paraphrase_low_threshold: float = Field(0.35, ge=0, le=1)

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    @property
    def max_request_bytes(self) -> int:
        """Upper bound on a whole multipart request (rejected before parsing)."""
        return self.max_upload_bytes * self.plagiarism_max_files + 1024 * 1024

    @property
    def cors_origins_list(self) -> list[str]:
        return [o.strip() for o in self.cors_allowed_origins.split(",") if o.strip()]

    @property
    def gemini_configured(self) -> bool:
        return bool(self.gemini_api_key and self.gemini_api_key.get_secret_value().strip())


@lru_cache
def get_settings() -> Settings:
    return Settings()
