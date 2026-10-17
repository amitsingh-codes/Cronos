"""Text normalisation, tokenising, sentence splitting and chunking (no NLTK downloads)."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

_WS = re.compile(r"\s+")
_WORD = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])|\n{2,}|\n(?=\s*(?:[-*•]|\d+[.)])\s)")
STOP_WORDS = frozenset(ENGLISH_STOP_WORDS)


def normalize_whitespace(text: str) -> str:
    return _WS.sub(" ", unicodedata.normalize("NFKC", text)).strip()


def tokenize(text: str, *, drop_stopwords: bool = True) -> list[str]:
    tokens = _WORD.findall(unicodedata.normalize("NFKC", text).lower())
    if drop_stopwords:
        tokens = [t for t in tokens if t not in STOP_WORDS and len(t) > 1]
    return tokens


def split_sentences(text: str) -> list[str]:
    text = unicodedata.normalize("NFKC", text).replace("\r", "\n")
    parts = _SENT_SPLIT.split(text)
    return [normalize_whitespace(p) for p in parts if p and normalize_whitespace(p)]


@dataclass(frozen=True)
class TextChunk:
    text: str
    page: int | None = None


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split ``text`` into windows of at most ``chunk_size`` chars with ``overlap`` shared chars.

    Cuts prefer paragraph/sentence/word boundaries. Always makes forward progress.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if not 0 <= overlap < chunk_size:
        raise ValueError("overlap must be >= 0 and smaller than chunk_size")
    text = text.strip()
    chunks: list[str] = []
    start, n = 0, len(text)
    while start < n:
        end = min(start + chunk_size, n)
        if end < n:
            window = text[start:end]
            floor = int(len(window) * 0.5)  # never cut in the first half of the window
            cut = max(window.rfind("\n\n"), window.rfind(". "), window.rfind("\n"), window.rfind(" "))
            if cut >= floor:
                end = start + cut + 1
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= n:
            break
        next_start = max(end - overlap, start + 1)
        # Snap forward to a word boundary so chunks do not begin mid-word.
        while 0 < next_start < n and not text[next_start - 1].isspace() and next_start < end:
            next_start += 1
        start = next_start
    return chunks


def truncate(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"
