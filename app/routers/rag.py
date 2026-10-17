import asyncio

from fastapi import APIRouter, Depends, File, Form, UploadFile
from starlette.concurrency import run_in_threadpool

from app.config import Settings
from app.dependencies import get_pdf_service, get_rag_service, get_settings_dep
from app.errors import AIConfigurationError, BadRequestError, RequestTimeoutError
from app.schemas.common import ErrorResponse
from app.schemas.rag import RagResponse
from app.services.pdf_service import PDFService
from app.services.rag_service import RagService
from app.utils.uploads import PDF_EXTS, read_upload, require_one

router = APIRouter(prefix="/api/rag", tags=["rag"])
ERRORS = {400: {"model": ErrorResponse}, 413: {"model": ErrorResponse}, 415: {"model": ErrorResponse},
          422: {"model": ErrorResponse}, 503: {"model": ErrorResponse}, 502: {"model": ErrorResponse}}


@router.post("/ask", response_model=RagResponse, responses=ERRORS)
async def ask(
    file: UploadFile = File(..., description="A PDF (text-based or scanned)."),
    question: str = Form(..., description="Question about the document."),
    top_k: int | None = Form(None, ge=1, le=10, description="Chunks to retrieve (default from settings)."),
    settings: Settings = Depends(get_settings_dep),
    pdf: PDFService = Depends(get_pdf_service),
    rag: RagService = Depends(get_rag_service),
) -> RagResponse:
    """Answer a question using only the content of the uploaded PDF. Nothing is stored."""
    question = require_one(question, "question")
    if len(question) > settings.rag_max_question_chars:
        raise BadRequestError(f"'question' exceeds {settings.rag_max_question_chars} characters.")
    if not rag.ai_configured:
        raise AIConfigurationError()
    upload = await read_upload(file, max_bytes=settings.max_upload_bytes, allowed=PDF_EXTS)
    k = top_k or settings.rag_top_k

    async def run() -> RagResponse:
        doc = await run_in_threadpool(pdf.extract, upload.data)
        chunks = await run_in_threadpool(rag.build_chunks, doc)
        hits = await run_in_threadpool(rag.retrieve, question, chunks, k)
        return await rag.answer(question, doc, k, chunks, hits)

    try:
        return await asyncio.wait_for(run(), timeout=settings.request_timeout_seconds)
    except asyncio.TimeoutError:
        raise RequestTimeoutError() from None
