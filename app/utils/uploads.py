"""Safe upload handling: bounded reads, extension + magic-byte checks, safe display names.

Uploaded bytes are processed in memory and never written to disk, so user-supplied
filenames are never used as paths. ``safe_name`` only produces a display label.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import PurePosixPath, PureWindowsPath

from fastapi import UploadFile

from app.errors import BadRequestError, CorruptFileError, FileTooLargeError, UnsupportedFileTypeError

PDF_EXTS = {".pdf"}
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class ValidatedUpload:
    name: str  # sanitised display name
    kind: str  # "pdf" | "image"
    data: bytes


def safe_name(raw: str | None, fallback: str = "upload") -> str:
    name = PureWindowsPath(PurePosixPath(raw or "").name).name  # drop any directory parts
    name = re.sub(r"[^A-Za-z0-9._ -]", "_", name).strip(" .")
    return name[:100] or fallback


def sniff_kind(data: bytes) -> str | None:
    if data.startswith(b"%PDF-"):
        return "pdf"
    if data.startswith((b"\x89PNG\r\n\x1a\n", b"\xff\xd8\xff", b"BM", b"II*\x00", b"MM\x00*")):
        return "image"
    return None


async def read_upload(
    file: UploadFile, *, max_bytes: int, allowed: set[str], index: int = 0
) -> ValidatedUpload:
    """Read an upload with a hard size cap and verify extension and real content type agree."""
    name = safe_name(file.filename, fallback=f"upload_{index + 1}")
    ext = "." + name.rsplit(".", 1)[-1].lower() if "." in name else ""
    if ext not in allowed:
        raise UnsupportedFileTypeError(
            f"'{name}': unsupported extension '{ext or 'none'}'. Allowed: {', '.join(sorted(allowed))}."
        )
    data = await file.read(max_bytes + 1)  # never buffer more than limit + 1 byte
    if len(data) > max_bytes:
        raise FileTooLargeError(f"'{name}' exceeds the {max_bytes // (1024 * 1024)} MB upload limit.")
    if not data:
        raise CorruptFileError(f"'{name}' is empty.")
    kind = sniff_kind(data)
    expected = "pdf" if ext in PDF_EXTS else "image"
    if kind != expected:
        raise UnsupportedFileTypeError(f"'{name}': file content does not match its '{ext}' extension.")
    return ValidatedUpload(name=name, kind=kind, data=data)


def require_one(value: str | None, field: str) -> str:
    if value is None or not value.strip():
        raise BadRequestError(f"'{field}' must not be empty.")
    return value.strip()
