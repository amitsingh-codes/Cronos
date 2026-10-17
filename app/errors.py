"""Application errors and the single, consistent error response format.

Every error body looks like::

    {"error": {"code": "machine_readable_code", "message": "Human readable.", "details": [...]}}

Messages are written to be safe to show to clients: they never include API keys,
prompts, raw model output, or student text.
"""

from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

logger = logging.getLogger(__name__)


class AppError(Exception):
    status_code = 500
    code = "internal_error"
    message = "Internal server error."

    def __init__(self, message: str | None = None, *, details: list | None = None) -> None:
        self.message = message or self.message
        self.details = details
        super().__init__(self.message)


class BadRequestError(AppError):
    status_code, code, message = 400, "bad_request", "Bad request."


class UnprocessableError(AppError):
    status_code, code, message = 422, "unprocessable", "The input could not be processed."


class UnsupportedFileTypeError(AppError):
    status_code, code, message = 415, "unsupported_file_type", "Unsupported file type."


class FileTooLargeError(AppError):
    status_code, code, message = 413, "file_too_large", "File is too large."


class EncryptedPDFError(UnprocessableError):
    code, message = "pdf_encrypted", "The PDF is password-protected and cannot be read."


class CorruptFileError(UnprocessableError):
    code, message = "file_corrupt", "The file is corrupt or not a valid document."


class NoTextExtractedError(UnprocessableError):
    code, message = "no_text_extracted", "No readable text could be extracted from the document."


class OCRUnavailableError(AppError):
    status_code, code = 503, "ocr_unavailable"
    message = "OCR is not available on this server (Tesseract missing, disabled, or failed)."


class AIConfigurationError(AppError):
    status_code, code = 503, "ai_not_configured"
    message = "The AI provider is not configured on this server (GEMINI_API_KEY is missing)."


class AIAuthenticationError(AppError):
    status_code, code = 503, "ai_authentication_failed"
    message = "The AI provider rejected the server's credentials."


class AIModelUnavailableError(AppError):
    status_code, code = 503, "ai_model_unavailable"
    message = "The configured AI model is unavailable. Check GEMINI_MODEL."


class AIRateLimitedError(AppError):
    status_code, code = 429, "ai_rate_limited"
    message = "The AI provider is rate-limiting or out of quota. Try again later."


class AITimeoutError(AppError):
    status_code, code = 504, "ai_timeout"
    message = "The AI provider timed out."


class AIUpstreamError(AppError):
    status_code, code = 502, "ai_upstream_error"
    message = "The AI provider returned an error."


class AIInvalidOutputError(AppError):
    status_code, code = 502, "ai_invalid_output"
    message = "The AI provider returned output that failed validation."


class RequestTimeoutError(AppError):
    status_code, code = 504, "request_timeout"
    message = "The request took too long to process."


def _body(code: str, message: str, details: list | None = None) -> dict:
    err: dict = {"code": code, "message": message}
    if details:
        err["details"] = details
    return {"error": err}


def register_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(AppError)
    async def _app_error(_: Request, exc: AppError) -> JSONResponse:
        return JSONResponse(_body(exc.code, exc.message, exc.details), status_code=exc.status_code)

    @app.exception_handler(RequestValidationError)
    async def _validation(_: Request, exc: RequestValidationError) -> JSONResponse:
        # Deliberately omit the submitted input: it may contain student data.
        details = [
            {"field": ".".join(str(p) for p in e["loc"] if p != "body"), "message": e["msg"]}
            for e in exc.errors()
        ]
        return JSONResponse(_body("validation_error", "Request validation failed.", details), status_code=422)

    @app.exception_handler(StarletteHTTPException)
    async def _http(_: Request, exc: StarletteHTTPException) -> JSONResponse:
        return JSONResponse(_body("http_error", str(exc.detail)), status_code=exc.status_code, headers=exc.headers)

    @app.exception_handler(Exception)
    async def _unhandled(_: Request, exc: Exception) -> JSONResponse:
        logger.exception("Unhandled error: %s", type(exc).__name__)
        return JSONResponse(_body("internal_error", "Internal server error."), status_code=500)
