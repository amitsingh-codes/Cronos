"""OCR via Tesseract (pytesseract). Tesseract itself must be installed separately."""

from __future__ import annotations

import io
import logging
import shutil

from PIL import Image, UnidentifiedImageError

from app.config import Settings
from app.errors import CorruptFileError, OCRUnavailableError

logger = logging.getLogger(__name__)


class OCRService:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        # Guard against decompression bombs: Pillow raises DecompressionBombError above this.
        Image.MAX_IMAGE_PIXELS = settings.ocr_max_image_pixels

    def available(self) -> bool:
        return self._settings.ocr_enabled and shutil.which("tesseract") is not None

    def image_to_text(self, image: Image.Image) -> str:
        if not self.available():
            raise OCRUnavailableError()
        try:
            import pytesseract

            return pytesseract.image_to_string(
                image.convert("RGB"),
                lang=self._settings.ocr_language,
                timeout=self._settings.ocr_timeout_seconds,
            )
        except OCRUnavailableError:
            raise
        except Exception as exc:  # pytesseract raises RuntimeError on timeout, TesseractError etc.
            logger.warning("OCR failed: %s", type(exc).__name__)
            raise OCRUnavailableError("OCR failed or timed out for this image.") from exc

    def image_bytes_to_text(self, data: bytes) -> str:
        try:
            with Image.open(io.BytesIO(data)) as img:
                img.load()
                return self.image_to_text(img)
        except (UnidentifiedImageError, Image.DecompressionBombError, OSError, SyntaxError) as exc:
            raise CorruptFileError("The image is corrupt or too large to process.") from exc
