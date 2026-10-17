"""FastAPI dependencies. Shared objects live on ``app.state`` (created once in ``create_app``)."""

from __future__ import annotations

from fastapi import Request

from app.config import Settings
from app.services.evaluation_service import EvaluationService
from app.services.ocr_service import OCRService
from app.services.pdf_service import PDFService
from app.services.plagiarism_service import PlagiarismService
from app.services.question_service import QuestionService
from app.services.rag_service import RagService


def get_settings_dep(request: Request) -> Settings:
    return request.app.state.settings


def get_ocr(request: Request) -> OCRService:
    return request.app.state.ocr


def get_pdf_service(request: Request) -> PDFService:
    return request.app.state.pdf


def get_rag_service(request: Request) -> RagService:
    return request.app.state.rag


def get_question_service(request: Request) -> QuestionService:
    return request.app.state.questions


def get_plagiarism_service(request: Request) -> PlagiarismService:
    return request.app.state.plagiarism


def get_evaluation_service(request: Request) -> EvaluationService:
    return request.app.state.evaluation
