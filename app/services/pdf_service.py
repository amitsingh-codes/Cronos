"""PDF text extraction with explicit handling of encrypted, corrupt and image-only PDFs."""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass

from pypdf import PdfReader
from pypdf.errors import PdfReadError, PyPdfError

from app.config import Settings
from app.errors import (
    CorruptFileError,
    EncryptedPDFError,
    NoTextExtractedError,
    OCRUnavailableError,
    UnprocessableError,
)
from app.services.ocr_service import OCRService

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PageText:
    number: int  # 1-based
    text: str


@dataclass(frozen=True)
class ExtractedDocument:
    pages: list[PageText]
    used_ocr: bool = False
    ocr_truncated: bool = False  # True when only the first ocr_max_pages pages were OCR'd

    @property
    def text(self) -> str:
        return "\n\n".join(p.text for p in self.pages if p.text.strip())


class PDFService:
    def __init__(self, settings: Settings, ocr: OCRService) -> None:
        self._settings = settings
        self._ocr = ocr

    def extract(self, data: bytes, *, allow_ocr: bool = True) -> ExtractedDocument:
        """Extract per-page text; fall back to OCR if the PDF has no selectable text."""
        reader = self._open(data)
        n_pages = len(reader.pages)
        if n_pages == 0:
            raise NoTextExtractedError("The PDF has no pages.")
        if n_pages > self._settings.pdf_max_pages:
            raise UnprocessableError(f"The PDF has {n_pages} pages; the limit is {self._settings.pdf_max_pages}.")

        pages: list[PageText] = []
        for i, page in enumerate(reader.pages, start=1):
            try:
                text = page.extract_text() or ""
            except Exception as exc:  # pypdf can raise many things on malformed pages
                logger.warning("pypdf failed on page %d: %s", i, type(exc).__name__)
                text = ""
            pages.append(PageText(i, text))

        if sum(len(p.text.strip()) for p in pages) >= self._settings.min_pdf_text_chars:
            return ExtractedDocument(pages)

        # No selectable text: probably a scan.
        if not allow_ocr or not self._ocr.available():
            raise NoTextExtractedError(
                "The PDF has no selectable text (likely scanned). OCR is disabled or unavailable "
                "on this server; install Tesseract or upload a text-based PDF."
            )
        return self._ocr_pdf(data, n_pages)

    # -- internals -------------------------------------------------------
    def _open(self, data: bytes) -> PdfReader:
        try:
            reader = PdfReader(io.BytesIO(data))
            if reader.is_encrypted:
                try:
                    ok = reader.decrypt("")  # PDFs with only an owner password open with ""
                except Exception:
                    ok = 0
                if not ok:
                    raise EncryptedPDFError()
            len(reader.pages)  # force parsing of the page tree to surface corruption now
            return reader
        except (EncryptedPDFError, UnprocessableError):
            raise
        except (PdfReadError, PyPdfError, ValueError, KeyError, TypeError, OSError, RecursionError) as exc:
            raise CorruptFileError("The PDF is corrupt or malformed.") from exc

    def _ocr_pdf(self, data: bytes, n_pages: int) -> ExtractedDocument:
        import pypdfium2 as pdfium

        limit = min(n_pages, self._settings.ocr_max_pages)
        scale = self._settings.ocr_dpi / 72
        pages: list[PageText] = []
        try:
            pdf = pdfium.PdfDocument(data)
            for i in range(limit):
                image = pdf[i].render(scale=scale).to_pil()
                pages.append(PageText(i + 1, self._ocr.image_to_text(image)))
        except OCRUnavailableError:
            raise
        except Exception as exc:
            raise CorruptFileError("The PDF pages could not be rendered for OCR.") from exc

        if not any(p.text.strip() for p in pages):
            raise NoTextExtractedError("OCR found no readable text in the PDF.")
        return ExtractedDocument(pages, used_ocr=True, ocr_truncated=n_pages > limit)
