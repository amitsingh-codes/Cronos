import asyncio

from fastapi import APIRouter, Depends, File, Form, UploadFile

from app.config import Settings
from app.dependencies import get_plagiarism_service, get_settings_dep
from app.errors import BadRequestError, RequestTimeoutError
from app.schemas.common import ErrorResponse
from app.schemas.plagiarism import PlagiarismResponse
from app.services.plagiarism_service import PlagiarismService
from app.utils.uploads import IMAGE_EXTS, PDF_EXTS, read_upload

router = APIRouter(prefix="/api/plagiarism", tags=["plagiarism"])


@router.post(
    "/check", response_model=PlagiarismResponse,
    responses={400: {"model": ErrorResponse}, 413: {"model": ErrorResponse}, 415: {"model": ErrorResponse},
               422: {"model": ErrorResponse}},
)
async def check(
    files: list[UploadFile] = File(..., description="Two or more PDFs / images (png, jpg, bmp, tiff)."),
    use_ai: bool = Form(False, description="Also ask Gemini for an (optional) paraphrase opinion."),
    include_excerpts: bool = Form(False, description="Include matching sentence excerpts in the response."),
    document_threshold: float | None = Form(None, ge=0, le=1, description="Override document similarity threshold."),
    sentence_threshold: float | None = Form(None, ge=0, le=1, description="Override sentence similarity threshold."),
    settings: Settings = Depends(get_settings_dep),
    svc: PlagiarismService = Depends(get_plagiarism_service),
) -> PlagiarismResponse:
    """Screen submissions for textual similarity. Flags are review signals, never proof of plagiarism."""
    if len(files) < 2:
        raise BadRequestError("Upload at least two files to compare.")
    if len(files) > settings.plagiarism_max_files:
        raise BadRequestError(f"At most {settings.plagiarism_max_files} files can be compared per request.")
    uploads = [
        await read_upload(f, max_bytes=settings.max_upload_bytes, allowed=PDF_EXTS | IMAGE_EXTS, index=i)
        for i, f in enumerate(files)
    ]
    thresholds = svc.thresholds(document_threshold, sentence_threshold)
    try:
        return await asyncio.wait_for(
            svc.check(uploads, use_ai=use_ai, include_excerpts=include_excerpts, thresholds=thresholds),
            timeout=settings.request_timeout_seconds,
        )
    except asyncio.TimeoutError:
        raise RequestTimeoutError() from None
