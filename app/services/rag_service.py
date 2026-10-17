"""Retrieval-augmented question answering over a single uploaded PDF (stateless, in-memory)."""

from __future__ import annotations

import logging

import numpy as np

from app.config import Settings
from app.errors import NoTextExtractedError, UnprocessableError
from app.schemas.rag import RagAnswerLLM, RagResponse, RagSource
from app.services.embedding_service import Embedder
from app.services.gemini_service import AIService
from app.services.pdf_service import ExtractedDocument
from app.utils.prompts import wrap_untrusted
from app.utils.text import TextChunk, chunk_text, truncate

logger = logging.getLogger(__name__)

SYSTEM = (
    "You answer questions strictly from the provided context excerpts of a document. "
    "If the context does not contain the information needed, set answerable=false and say so briefly; "
    "do not use outside knowledge and do not guess. Cite page numbers inline like (p. 3) when pages are given. "
    "Respond as JSON with fields: answerable (boolean) and answer (string)."
)


class RagService:
    def __init__(self, settings: Settings, ai: AIService, embedder: Embedder) -> None:
        self._s = settings
        self._ai = ai
        self._embedder = embedder

    @property
    def ai_configured(self) -> bool:
        return self._ai.is_configured

    # -- pure/sync steps (run in a worker thread by the router) --------------
    def build_chunks(self, doc: ExtractedDocument) -> list[TextChunk]:
        chunks: list[TextChunk] = []
        for page in doc.pages:
            for piece in chunk_text(page.text, self._s.rag_chunk_size, self._s.rag_chunk_overlap):
                chunks.append(TextChunk(piece, page.number))
        if not chunks:
            raise NoTextExtractedError()
        if len(chunks) > self._s.rag_max_chunks:
            raise UnprocessableError(
                f"The document produced {len(chunks)} chunks; the limit is {self._s.rag_max_chunks}. "
                "Upload a shorter document."
            )
        return chunks

    def retrieve(self, question: str, chunks: list[TextChunk], top_k: int) -> list[tuple[int, float]]:
        """Return [(chunk_index, cosine_score)] best-first. Safe for tiny documents."""
        vectors = self._embedder.encode([c.text for c in chunks] + [question])
        doc_vecs, q_vec = vectors[:-1], vectors[-1]
        if doc_vecs.ndim != 2 or doc_vecs.shape[0] != len(chunks) or doc_vecs.shape[1] != q_vec.shape[0]:
            raise UnprocessableError("Embedding dimensions were inconsistent; cannot search this document.")
        scores = doc_vecs @ q_vec  # vectors are L2-normalised => cosine similarity
        k = min(top_k, len(chunks))
        order = np.argsort(-scores)[:k]
        return [(int(i), float(scores[i])) for i in order]

    # -- async orchestration ---------------------------------------------------
    async def answer(
        self, question: str, doc: ExtractedDocument, top_k: int | None, chunks: list[TextChunk],
        hits: list[tuple[int, float]],
    ) -> RagResponse:
        context = "\n\n".join(
            f"[chunk {i} | page {chunks[i].page}]\n{chunks[i].text}" for i, _ in hits
        )
        prompt = (
            f"{wrap_untrusted('context', context)}\n\n"
            f"Question: {question}\n\nAnswer using only the context above."
        )
        out = await self._ai.generate_json(prompt=prompt, schema=RagAnswerLLM, system_instruction=SYSTEM)
        sources = [
            RagSource(chunk_id=i, page=chunks[i].page, score=round(s, 4), excerpt=truncate(chunks[i].text, 300))
            for i, s in hits
        ]
        return RagResponse(
            answer=out.answer.strip(),
            answerable=out.answerable,
            sources=sources,
            metadata={
                "pages": len(doc.pages),
                "chunks_indexed": len(chunks),
                "chunks_retrieved": len(hits),
                "used_ocr": doc.used_ocr,
                "ocr_truncated": doc.ocr_truncated,
                "model": self._s.gemini_model,
            },
        )
