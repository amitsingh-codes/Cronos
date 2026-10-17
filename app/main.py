"""Application factory. ``create_app`` accepts overrides so tests can inject fakes."""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import Settings, get_settings
from app.errors import register_error_handlers
from app.middleware import RateLimitMiddleware, RequestSizeLimitMiddleware
from app.routers import evaluation, health, plagiarism, questions, rag
from app.services.embedding_service import Embedder, SentenceTransformerEmbedder
from app.services.evaluation_service import EvaluationService
from app.services.gemini_service import AIService, GeminiService
from app.services.ocr_service import OCRService
from app.services.pdf_service import PDFService
from app.services.plagiarism_service import PlagiarismService
from app.services.question_service import QuestionService
from app.services.rag_service import RagService

logger = logging.getLogger(__name__)


def create_app(
    settings: Settings | None = None,
    *,
    ai: AIService | None = None,
    embedder: Embedder | None = None,
    ocr: OCRService | None = None,
) -> FastAPI:
    settings = settings or get_settings()
    logging.basicConfig(level=settings.log_level.upper(), format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if settings.preload_embedding_model and hasattr(app.state.embedder, "load"):
            await asyncio.to_thread(app.state.embedder.load)
        yield

    app = FastAPI(
        title=settings.app_name, version=settings.app_version, lifespan=lifespan,
        description="Backend for ScoreMatrix: RAG Q&A, question generation, plagiarism screening and rubric-based evaluation. "
                    "Stateless: nothing is stored. AI output is advisory and must be reviewed by a teacher.",
    )

    ai = ai or GeminiService(settings)
    embedder = embedder or SentenceTransformerEmbedder(settings)
    ocr = ocr or OCRService(settings)
    pdf = PDFService(settings, ocr)

    app.state.settings = settings
    app.state.ai, app.state.embedder, app.state.ocr, app.state.pdf = ai, embedder, ocr, pdf
    app.state.rag = RagService(settings, ai, embedder)
    app.state.questions = QuestionService(settings, ai)
    app.state.evaluation = EvaluationService(settings, ai)
    app.state.plagiarism = PlagiarismService(settings, ai, pdf, ocr)

    register_error_handlers(app)
    app.add_middleware(RateLimitMiddleware, per_minute=settings.rate_limit_per_minute)
    app.add_middleware(RequestSizeLimitMiddleware, max_bytes=settings.max_request_bytes)
    if settings.cors_origins_list:
        app.add_middleware(
            CORSMiddleware, allow_origins=settings.cors_origins_list, allow_methods=["GET", "POST"],
            allow_headers=["Content-Type", "Authorization"],
        )

    for r in (health, rag, questions, plagiarism, evaluation):
        app.include_router(r.router)
    return app


def _default_app() -> FastAPI:
    return create_app()


app = _default_app()
