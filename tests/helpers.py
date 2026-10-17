"""Test helpers: fake Gemini/embedder/OCR and tiny PDF/image builders (no network needed)."""

from __future__ import annotations

import hashlib
import io
import re

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pypdf import PdfWriter

from app.config import Settings
from app.services.ocr_service import OCRService
from app.errors import OCRUnavailableError


def make_settings(**overrides) -> Settings:
    base = dict(gemini_api_key="test-key-not-real", rate_limit_per_minute=0, gemini_backoff_base_seconds=0,
                ocr_timeout_seconds=20)
    base.update(overrides)
    return Settings(_env_file=None, **base)


class FakeAI:
    """Replays queued responses. Items: dict/BaseModel -> validated; Exception -> raised."""

    def __init__(self, responses=None, configured: bool = True) -> None:
        self.responses = list(responses or [])
        self.configured = configured
        self.calls: list[dict] = []

    @property
    def is_configured(self) -> bool:
        return self.configured

    async def generate_json(self, *, prompt, schema, system_instruction, temperature=0.2):
        self.calls.append({"prompt": prompt, "schema": schema, "system": system_instruction})
        if not self.configured:
            from app.errors import AIConfigurationError
            raise AIConfigurationError()
        item = self.responses.pop(0) if len(self.responses) > 1 else self.responses[0]
        if isinstance(item, Exception):
            raise item
        return item if isinstance(item, schema) else schema.model_validate(item)


class FakeEmbedder:
    """Deterministic bag-of-words hashing embedder: similar words => similar vectors."""

    dim = 64

    def encode(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for r, t in enumerate(texts):
            for w in re.findall(r"[a-z0-9]+", t.lower()):
                out[r, int(hashlib.md5(w.encode()).hexdigest(), 16) % self.dim] += 1
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms[norms == 0] = 1
        return out / norms


class FakeOCR(OCRService):
    def __init__(self, settings: Settings, text: str = "ocr text", available: bool = True, fail: bool = False):
        super().__init__(settings)
        self.text, self._available, self.fail = text, available, fail

    def available(self) -> bool:
        return self._available

    def image_to_text(self, image) -> str:
        if self.fail or not self._available:
            raise OCRUnavailableError()
        return self.text


def make_text_pdf(text: str, per_page: int = 40) -> bytes:
    """Build a minimal valid text PDF (Helvetica) by hand."""
    words, lines, cur = text.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > 90:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    lines.append(cur)
    pages = [lines[i:i + per_page] for i in range(0, len(lines), per_page)] or [[""]]

    objs: list[bytes] = []
    n_pages = len(pages)
    kids = " ".join(f"{4 + 2 * i} 0 R" for i in range(n_pages))
    objs.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    objs.append(f"<< /Type /Pages /Kids [{kids}] /Count {n_pages} >>".encode())
    objs.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    for i, pl in enumerate(pages):
        esc = lambda s: s.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        stream = "BT /F1 11 Tf 50 750 Td 14 TL\n" + "\n".join(f"({esc(l)}) Tj T*" for l in pl) + "\nET"
        objs.append(f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents {5 + 2 * i} 0 R "
                    f"/Resources << /Font << /F1 3 0 R >> >> >>".encode())
        objs.append(f"<< /Length {len(stream)} >>\nstream\n{stream}\nendstream".encode())
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for i, o in enumerate(objs, 1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n".encode() + o + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode()
    for off in offsets:
        out += f"{off:010d} 00000 n \n".encode()
    out += f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF".encode()
    return bytes(out)


def make_blank_pdf() -> bytes:
    w = PdfWriter()
    w.add_blank_page(612, 792)
    buf = io.BytesIO()
    w.write(buf)
    return buf.getvalue()


def make_encrypted_pdf(password: str = "secret") -> bytes:
    w = PdfWriter(clone_from=io.BytesIO(make_text_pdf("hello world " * 20)))
    w.encrypt(password)
    buf = io.BytesIO()
    w.write(buf)
    return buf.getvalue()


def make_text_image(text: str, fmt: str = "PNG", size=(1400, 220), font_size: int = 60) -> bytes:
    img = Image.new("RGB", size, "white")
    ImageDraw.Draw(img).text((20, 60), text, fill="black", font=ImageFont.load_default(size=font_size))
    buf = io.BytesIO()
    img.save(buf, fmt)
    return buf.getvalue()


def make_image_only_pdf(text: str = "Scanned page text") -> bytes:
    img = Image.open(io.BytesIO(make_text_image(text)))
    buf = io.BytesIO()
    img.save(buf, "PDF")
    return buf.getvalue()
