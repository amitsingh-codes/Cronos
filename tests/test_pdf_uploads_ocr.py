import shutil

import pytest

from app.errors import CorruptFileError, EncryptedPDFError, NoTextExtractedError, OCRUnavailableError
from app.services.ocr_service import OCRService
from app.services.pdf_service import PDFService
from app.utils.text import chunk_text
from app.utils.uploads import safe_name, sniff_kind
from tests.helpers import (FakeOCR, make_blank_pdf, make_encrypted_pdf, make_image_only_pdf, make_settings,
                           make_text_image, make_text_pdf)

S = make_settings()


def pdf_service(ocr=None):
    return PDFService(S, ocr or FakeOCR(S, available=False))


def test_extracts_text_per_page():
    text = " ".join(f"word{i}" for i in range(400))
    doc = pdf_service().extract(make_text_pdf(text, per_page=10))
    assert len(doc.pages) > 1 and doc.used_ocr is False
    assert "word0" in doc.pages[0].text and "word399" in doc.text


def test_corrupt_pdf():
    with pytest.raises(CorruptFileError):
        pdf_service().extract(b"%PDF-1.4\nthis is garbage \x00\x01")


def test_truncated_pdf_is_corrupt_not_crash():
    with pytest.raises(CorruptFileError):
        pdf_service().extract(make_text_pdf("hello " * 100)[:200])


def test_encrypted_pdf():
    with pytest.raises(EncryptedPDFError):
        pdf_service().extract(make_encrypted_pdf())


def test_empty_pdf_without_ocr_gives_clear_error():
    with pytest.raises(NoTextExtractedError) as ei:
        pdf_service().extract(make_blank_pdf())
    assert "OCR" in ei.value.message


def test_image_only_pdf_uses_ocr_fallback():
    svc = pdf_service(FakeOCR(S, text="Scanned content from OCR"))
    doc = svc.extract(make_image_only_pdf())
    assert doc.used_ocr and "Scanned content" in doc.text


def test_ocr_fallback_disabled_flag():
    with pytest.raises(NoTextExtractedError):
        pdf_service(FakeOCR(S)).extract(make_image_only_pdf(), allow_ocr=False)


def test_ocr_returning_nothing_is_no_text():
    with pytest.raises(NoTextExtractedError):
        pdf_service(FakeOCR(S, text="   ")).extract(make_image_only_pdf())


def test_ocr_failure_propagates_as_ocr_unavailable():
    with pytest.raises(OCRUnavailableError):
        pdf_service(FakeOCR(S, fail=True)).extract(make_image_only_pdf())


def test_page_limit_enforced():
    from app.errors import UnprocessableError
    svc = PDFService(make_settings(pdf_max_pages=1), FakeOCR(S))
    with pytest.raises(UnprocessableError):
        svc.extract(make_text_pdf("word " * 3000, per_page=10))


# --- utilities -------------------------------------------------------------------
def test_safe_name_strips_paths_and_junk():
    assert safe_name("../../etc/passwd") == "passwd"
    assert safe_name("C:\\Users\\x\\ans<>.pdf") == "ans__.pdf"
    assert safe_name(None, "fallback") == "fallback"
    assert safe_name("....") == "upload"


def test_sniff_kind():
    assert sniff_kind(b"%PDF-1.7") == "pdf" and sniff_kind(b"\x89PNG\r\n\x1a\n") == "image"
    assert sniff_kind(b"MZ\x90\x00") is None


def test_chunking_overlap_progress_and_bounds():
    text = ("Sentence number %d is here. " * 1) * 1
    text = " ".join(f"Sentence number {i} is here." for i in range(200))
    chunks = chunk_text(text, 200, 40)
    assert len(chunks) > 5 and all(len(c) <= 200 for c in chunks)
    assert chunks[0][-20:].split()[-1] in chunks[1]  # overlap exists
    assert chunk_text("", 100, 10) == [] and chunk_text("short", 100, 10) == ["short"]
    with pytest.raises(ValueError):
        chunk_text("abc", 10, 10)


def test_chunking_unbroken_text_terminates():
    assert len("".join(chunk_text("x" * 5000, 100, 20))) >= 5000 - 100


# --- REAL Tesseract (skipped when not installed) -----------------------------------
needs_tesseract = pytest.mark.skipif(shutil.which("tesseract") is None, reason="tesseract not installed")


@needs_tesseract
def test_real_ocr_reads_image():
    ocr = OCRService(make_settings())
    text = ocr.image_bytes_to_text(make_text_image("Photosynthesis converts light", "PNG"))
    assert "photosynthesis" in text.lower()


@needs_tesseract
def test_real_ocr_scanned_pdf_end_to_end():
    svc = PDFService(S, OCRService(make_settings()))
    doc = svc.extract(make_image_only_pdf("Mitochondria produce energy"))
    assert doc.used_ocr and "mitochondria" in doc.text.lower()


def test_corrupt_image_bytes_raise_corrupt():
    with pytest.raises(CorruptFileError):
        OCRService(make_settings()).image_bytes_to_text(b"\x89PNG\r\n\x1a\nnotreallyapng")


def test_ocr_unavailable_raises(monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda _: None)
    from PIL import Image
    with pytest.raises(OCRUnavailableError):
        OCRService(make_settings()).image_to_text(Image.new("RGB", (10, 10)))
