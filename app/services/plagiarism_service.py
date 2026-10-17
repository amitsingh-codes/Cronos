"""Plagiarism *screening*: document TF-IDF similarity + sentence overlap + optional Gemini paraphrase opinion.

Output is a review signal only. It never states that a student plagiarised.
"""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass
from itertools import combinations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.config import Settings
from app.errors import AppError, NoTextExtractedError
from app.schemas.plagiarism import (
    DocumentInfo,
    MatchingSentencePair,
    PairResult,
    ParaphraseAssessment,
    ParaphraseLLM,
    PlagiarismResponse,
    SentenceOverlap,
    SkippedDocument,
)
from app.services.gemini_service import AIService
from app.services.ocr_service import OCRService
from app.services.pdf_service import PDFService
from app.utils.prompts import wrap_untrusted
from app.utils.text import split_sentences, tokenize, truncate
from app.utils.uploads import ValidatedUpload

logger = logging.getLogger(__name__)
MIN_UNIQUE_TOKENS = 5
_PUNCT = re.compile(r"[^\w\s]")

SYSTEM = (
    "You assist a teacher with plagiarism screening. You are shown sentence pairs from two student submissions that are "
    "moderately similar. Judge ONLY whether the pairs suggest one text may be a paraphrase of the other rather than "
    "coincidence caused by a shared topic, textbook facts or common phrasing. You cannot prove copying; be conservative "
    "and lower your confidence when the overlap could be explained by common knowledge. Return JSON only."
)


@dataclass(frozen=True)
class Thresholds:
    document: float
    sentence: float
    min_matching_sentences: int
    sentence_ratio: float
    min_sentence_tokens: int
    paraphrase_low: float

    def as_dict(self) -> dict:
        return {
            "document_similarity": self.document,
            "sentence_similarity": self.sentence,
            "min_matching_sentences": self.min_matching_sentences,
            "min_sentence_overlap_ratio": self.sentence_ratio,
            "min_words_per_sentence": self.min_sentence_tokens,
        }


@dataclass
class _Doc:
    name: str
    text: str
    used_ocr: bool
    tokens: list[str]
    sentences: list[str]
    sentence_tokens: list[list[str]]

    @property
    def usable(self) -> bool:
        return len(self.tokens) >= 1 and len(set(self.tokens)) >= MIN_UNIQUE_TOKENS


def _norm_sentence(s: str) -> str:
    return re.sub(r"\s+", " ", _PUNCT.sub("", s.lower())).strip()


class PlagiarismService:
    def __init__(self, settings: Settings, ai: AIService, pdf: PDFService, ocr: OCRService) -> None:
        self._s = settings
        self._ai = ai
        self._pdf = pdf
        self._ocr = ocr

    def thresholds(self, document: float | None = None, sentence: float | None = None) -> Thresholds:
        s = self._s
        return Thresholds(
            document=s.plagiarism_doc_threshold if document is None else document,
            sentence=s.plagiarism_sentence_threshold if sentence is None else sentence,
            min_matching_sentences=s.plagiarism_min_matching_sentences,
            sentence_ratio=s.plagiarism_sentence_ratio_threshold,
            min_sentence_tokens=s.plagiarism_min_sentence_tokens,
            paraphrase_low=s.plagiarism_paraphrase_low_threshold,
        )

    # ------------------------------------------------------------------ public
    async def check(
        self, uploads: list[ValidatedUpload], *, use_ai: bool, include_excerpts: bool, thresholds: Thresholds
    ) -> PlagiarismResponse:
        names = self._unique_names([u.name for u in uploads])
        extracted = await asyncio.gather(*(asyncio.to_thread(self._extract, u) for u in uploads))

        docs: list[_Doc] = []
        skipped: list[SkippedDocument] = []
        for name, res in zip(names, extracted):
            if isinstance(res, str):
                skipped.append(SkippedDocument(name=name, reason=res))
            else:
                text, used_ocr = res
                docs.append(self._prepare(name, text, used_ocr))

        if not docs:
            raise NoTextExtractedError(
                "No readable text could be extracted from any uploaded file.",
                details=[s.model_dump() for s in skipped],
            )

        result, candidates = await asyncio.to_thread(self._analyse, docs, skipped, include_excerpts, thresholds)
        if use_ai and result.pairs:
            await self._add_ai_assessments(result, candidates, thresholds)
        return result

    # --------------------------------------------------------------- extraction
    def _extract(self, upload: ValidatedUpload) -> tuple[str, bool] | str:
        """Return (text, used_ocr) or a human-readable skip reason."""
        try:
            if upload.kind == "pdf":
                doc = self._pdf.extract(upload.data)
                return doc.text, doc.used_ocr
            return self._ocr.image_bytes_to_text(upload.data), True
        except AppError as exc:
            return exc.message
        except Exception as exc:
            logger.warning("Unexpected extraction failure: %s", type(exc).__name__)
            return "The file could not be processed."

    @staticmethod
    def _unique_names(names: list[str]) -> list[str]:
        seen: dict[str, int] = {}
        out = []
        for n in names:
            seen[n] = seen.get(n, 0) + 1
            out.append(n if seen[n] == 1 else f"{n} ({seen[n]})")
        return out

    def _prepare(self, name: str, text: str, used_ocr: bool) -> _Doc:
        sentences = split_sentences(text)[: self._s.plagiarism_max_sentences_per_doc]
        return _Doc(
            name=name, text=text, used_ocr=used_ocr, tokens=tokenize(text), sentences=sentences,
            sentence_tokens=[tokenize(s) for s in sentences],
        )

    # ------------------------------------------------------------------ analysis
    def _analyse(self, docs: list[_Doc], skipped, include_excerpts: bool, th: Thresholds):
        warnings: list[str] = []
        infos = [
            DocumentInfo(
                name=d.name, characters=len(d.text), meaningful_words=len(d.tokens), used_ocr=d.used_ocr,
                status="compared" if d.usable else "insufficient_text",
            )
            for d in docs
        ]
        usable = [d for d in docs if d.usable]
        for d in docs:
            if not d.usable:
                warnings.append(f"'{d.name}' has too little meaningful text (needs at least {MIN_UNIQUE_TOKENS} distinct words) and was not compared.")
            elif len(d.tokens) < self._s.plagiarism_min_tokens:
                warnings.append(f"'{d.name}' is very short ({len(d.tokens)} meaningful words); similarity scores are unreliable.")
        if any(d.used_ocr for d in docs):
            warnings.append("Some text came from OCR; recognition errors can lower similarity scores (false negatives).")

        pairs: list[PairResult] = []
        candidates: dict[tuple[str, str], list[tuple[str, str, float]]] = {}
        if len(usable) < 2:
            warnings.append("Fewer than two documents had enough text to compare.")
        else:
            matrix = self._doc_similarity_matrix(usable)
            for i, j in combinations(range(len(usable)), 2):
                a, b = usable[i], usable[j]
                overlap, cands = self._sentence_overlap(a, b, th, include_excerpts)
                sim = float(matrix[i, j])
                reasons = self._flag_reasons(sim, overlap, th)
                pairs.append(PairResult(
                    document_a=a.name, document_b=b.name, document_similarity=round(sim, 4),
                    document_similarity_percent=round(sim * 100, 2), sentence_overlap=overlap,
                    flagged_for_review=bool(reasons), flag_reasons=reasons,
                ))
                candidates[(a.name, b.name)] = cands
            pairs.sort(key=lambda p: (p.flagged_for_review, p.document_similarity), reverse=True)

        return (
            PlagiarismResponse(documents=infos, skipped_documents=list(skipped), pairs=pairs,
                               thresholds=th.as_dict(), warnings=warnings),
            candidates,
        )

    @staticmethod
    def _doc_similarity_matrix(docs: list[_Doc]) -> np.ndarray:
        vec = TfidfVectorizer(analyzer=str.split, sublinear_tf=True)
        tfidf = vec.fit_transform([" ".join(d.tokens) for d in docs])
        return cosine_similarity(tfidf)

    @staticmethod
    def _flag_reasons(sim: float, o: SentenceOverlap, th: Thresholds) -> list[str]:
        reasons = []
        if sim >= th.document:
            reasons.append(f"Document-level TF-IDF cosine similarity {sim * 100:.1f}% is at or above the {th.document * 100:.0f}% threshold.")
        matched = max(o.matched_a, o.matched_b)
        ratio = max(o.matched_ratio_a, o.matched_ratio_b)
        if matched >= th.min_matching_sentences and ratio >= th.sentence_ratio:
            reasons.append(
                f"{o.matched_a} sentence(s) in document A ({o.matched_ratio_a * 100:.0f}%) and {o.matched_b} in B "
                f"({o.matched_ratio_b * 100:.0f}%) closely match sentences in the other document "
                f"(similarity ≥ {th.sentence:.2f}; at least {th.min_matching_sentences} matches and "
                f"{th.sentence_ratio * 100:.0f}% overlap required)."
            )
        return reasons

    def _sentence_overlap(self, a: _Doc, b: _Doc, th: Thresholds, include_excerpts: bool):
        idx_a = [i for i, t in enumerate(a.sentence_tokens) if len(t) >= 1 and len(a.sentences[i].split()) >= th.min_sentence_tokens]
        idx_b = [i for i, t in enumerate(b.sentence_tokens) if len(t) >= 1 and len(b.sentences[i].split()) >= th.min_sentence_tokens]
        empty = SentenceOverlap(
            sentences_compared_a=len(idx_a), sentences_compared_b=len(idx_b), matched_a=0, matched_b=0,
            matched_ratio_a=0.0, matched_ratio_b=0.0, exact_matches=0, top_matches=[] if include_excerpts else None,
        )
        if not idx_a or not idx_b:
            return empty, []

        sa = [a.sentences[i] for i in idx_a]
        sb = [b.sentences[i] for i in idx_b]
        vec = TfidfVectorizer(analyzer=str.split)
        vec.fit([" ".join(a.sentence_tokens[i]) for i in idx_a] + [" ".join(b.sentence_tokens[i]) for i in idx_b])
        ma = vec.transform([" ".join(a.sentence_tokens[i]) for i in idx_a])
        mb = vec.transform([" ".join(b.sentence_tokens[i]) for i in idx_b])
        sim = cosine_similarity(ma, mb)  # (len_a, len_b)

        hit = sim >= th.sentence
        matched_a, matched_b = int(hit.any(axis=1).sum()), int(hit.any(axis=0).sum())
        exact = len({_norm_sentence(s) for s in sa} & {_norm_sentence(s) for s in sb})

        top: list[MatchingSentencePair] | None = None
        if include_excerpts:
            flat = [(float(sim[i, j]), i, j) for i, j in zip(*np.where(hit))]
            flat.sort(reverse=True)
            top = [MatchingSentencePair(similarity=round(s, 3), excerpt_a=truncate(sa[i], 200), excerpt_b=truncate(sb[j], 200))
                   for s, i, j in flat[:5]]

        # Moderately similar pairs (below the "copied" threshold) are paraphrase candidates for Gemini.
        cands: list[tuple[str, str, float]] = []
        best_j = sim.argmax(axis=1)
        for i in range(sim.shape[0]):
            s = float(sim[i, best_j[i]])
            if th.paraphrase_low <= s < th.sentence:
                cands.append((sa[i], sb[best_j[i]], s))
        cands.sort(key=lambda c: c[2], reverse=True)

        return SentenceOverlap(
            sentences_compared_a=len(sa), sentences_compared_b=len(sb), matched_a=matched_a, matched_b=matched_b,
            matched_ratio_a=round(matched_a / len(sa), 3), matched_ratio_b=round(matched_b / len(sb), 3),
            exact_matches=exact, top_matches=top,
        ), cands[:5]

    # ------------------------------------------------------------------ Gemini
    async def _add_ai_assessments(self, result: PlagiarismResponse, candidates, th: Thresholds) -> None:
        """Ask Gemini about pairs that are flagged OR have moderately-similar sentence pairs.

        Paraphrases score low on TF-IDF, so eligibility is based on sentence-level candidates,
        not on document similarity alone. ``result.pairs`` is already sorted most-suspicious first.
        """
        def has_signal(p: PairResult) -> bool:
            return p.flagged_for_review or bool(candidates.get((p.document_a, p.document_b)))

        eligible = [p for p in result.pairs if has_signal(p)]
        chosen = {id(p) for p in eligible[: self._s.plagiarism_ai_max_pairs]}
        if len(eligible) > len(chosen):
            result.warnings.append(f"AI paraphrase assessment was limited to the top {self._s.plagiarism_ai_max_pairs} pairs.")
        for pair in result.pairs:
            if id(pair) not in chosen:
                note = "Pair limit reached." if has_signal(pair) else "No moderately similar sentence pairs to assess."
                pair.paraphrase_assessment = ParaphraseAssessment(status="skipped", note=f"Not assessed: {note}")
            else:
                pair.paraphrase_assessment = await self._assess(candidates.get((pair.document_a, pair.document_b), []))

    async def _assess(self, cands: list[tuple[str, str, float]]) -> ParaphraseAssessment:
        if not cands:
            return ParaphraseAssessment(status="skipped", note="No moderately similar sentence pairs were available to assess.")
        if not self._ai.is_configured:
            return ParaphraseAssessment(status="unavailable", note="Gemini is not configured on this server.")
        body = "\n\n".join(f"Pair {i}:\nA: {a}\nB: {b}" for i, (a, b, _) in enumerate(cands, 1))
        try:
            out = await self._ai.generate_json(
                prompt=wrap_untrusted("document", body) + "\n\nAssess whether these pairs suggest paraphrasing.",
                schema=ParaphraseLLM, system_instruction=SYSTEM, temperature=0.0,
            )
        except AppError as exc:
            return ParaphraseAssessment(status="unavailable", note=f"AI assessment failed: {exc.message}")
        return ParaphraseAssessment(
            status="completed", likely_paraphrase=out.likely_paraphrase,
            confidence=round(min(max(out.confidence, 0.0), 1.0), 2), explanation=truncate(out.explanation.strip(), 600),
            note=f"Based on {len(cands)} sentence pair(s) only.",
        )
