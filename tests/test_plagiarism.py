from app.errors import AIUpstreamError
from tests.helpers import FakeAI, FakeOCR, make_settings, make_text_image, make_text_pdf

PHOTO = (
    "Photosynthesis is the process by which green plants convert sunlight into chemical energy stored in glucose. "
    "The reaction takes place inside chloroplasts, which contain the green pigment chlorophyll. "
    "Plants absorb carbon dioxide from the air through tiny pores called stomata. "
    "Water is taken up by the roots and transported to the leaves through the xylem vessels. "
    "Oxygen is released as a by-product of the light dependent reactions during the day. "
    "The glucose produced is used for respiration or converted into starch for long term storage."
)
PHOTO_COPY = PHOTO.replace("green plants", "all green plants").replace("tiny pores", "small pores")
PHOTO_PARAPHRASE = (
    "Green vegetation transforms solar radiation into stored sugar through a biochemical pathway known as photosynthesis. "
    "Chloroplasts, rich in chlorophyll pigment, are the organelles hosting this reaction. "
    "Atmospheric carbon dioxide enters leaves via microscopic openings named stomata. "
    "Roots draw up water which travels toward the foliage inside xylem tubes. "
    "Gaseous oxygen leaves the plant as a secondary product of daytime light reactions. "
    "Sugar made this way fuels respiration or becomes starch kept for later."
)
REVOLUTION = (
    "The French Revolution began in 1789 when financial crisis and social inequality pushed the Third Estate to rebel. "
    "The storming of the Bastille became a powerful symbol of resistance against royal authority. "
    "The Declaration of the Rights of Man proclaimed liberty and equality as fundamental principles. "
    "Radical factions such as the Jacobins eventually seized control and launched the Reign of Terror. "
    "Napoleon Bonaparte rose to power after the chaos and crowned himself emperor in 1804. "
    "Historians still debate how much the revolution changed everyday life for ordinary peasants."
)


def pdf(text):
    return make_text_pdf(text)


def check(c, docs: dict, **form):
    files = [("files", (n, d, "application/pdf" if d.startswith(b"%PDF") else "image/png")) for n, d in docs.items()]
    return c.post("/api/plagiarism/check", files=files, data={k: str(v).lower() if isinstance(v, bool) else v for k, v in form.items()})


def pair(body, a, b):
    return next(p for p in body["pairs"] if {p["document_a"], p["document_b"]} == {a, b})


def test_near_copy_is_flagged_with_explained_signals(build_client):
    r = check(build_client(), {"a.pdf": pdf(PHOTO), "b.pdf": pdf(PHOTO_COPY), "c.pdf": pdf(REVOLUTION)})
    assert r.status_code == 200, r.text
    d = r.json()
    ab = pair(d, "a.pdf", "b.pdf")
    assert ab["flagged_for_review"] and ab["document_similarity_percent"] > 75
    assert ab["sentence_overlap"]["matched_a"] >= 5 and ab["sentence_overlap"]["matched_ratio_a"] > 0.8
    assert len(ab["flag_reasons"]) == 2 and "TF-IDF" in ab["flag_reasons"][0]
    assert d["pairs"][0] is not None and {d["pairs"][0]["document_a"], d["pairs"][0]["document_b"]} == {"a.pdf", "b.pdf"}  # most suspicious first
    assert "not proof" in d["disclaimer"].lower() and d["thresholds"]["document_similarity"] == 0.75


def test_dissimilar_documents_not_flagged(build_client):
    d = check(build_client(), {"a.pdf": pdf(PHOTO), "c.pdf": pdf(REVOLUTION)}).json()
    p = d["pairs"][0]
    assert not p["flagged_for_review"] and p["flag_reasons"] == []
    assert p["document_similarity"] < 0.2 and p["sentence_overlap"]["matched_a"] == 0


def test_excerpts_hidden_by_default_and_opt_in(build_client):
    docs = {"a.pdf": pdf(PHOTO), "b.pdf": pdf(PHOTO_COPY)}
    c = build_client()
    assert check(c, docs).json()["pairs"][0]["sentence_overlap"]["top_matches"] is None
    top = check(c, docs, include_excerpts=True).json()["pairs"][0]["sentence_overlap"]["top_matches"]
    assert top and len(top) <= 5 and "excerpt_a" in top[0]


def test_thresholds_configurable_per_request_and_env(build_client):
    docs = {"a.pdf": pdf(PHOTO), "b.pdf": pdf(PHOTO_PARAPHRASE)}
    base = check(build_client(), docs).json()["pairs"][0]["document_similarity"]
    strict = check(build_client(), docs, document_threshold=0.0).json()["pairs"][0]
    assert strict["flagged_for_review"] and strict["flag_reasons"]
    lenient = check(build_client(plagiarism_doc_threshold=1.0), {"a.pdf": pdf(PHOTO), "b.pdf": pdf(PHOTO_COPY)}).json()["pairs"][0]
    assert base < 0.75 and lenient["document_similarity"] > 0.75  # would trip the default threshold...
    assert not any("TF-IDF" in r for r in lenient["flag_reasons"])   # ...but not at threshold 1.0
    assert check(build_client(), docs, document_threshold=1.5).status_code == 422


def test_empty_and_punctuation_only_documents_do_not_crash(build_client):
    docs = {"a.pdf": pdf(PHOTO), "punct.pdf": pdf("!!! ??? ... --- ;;; ,,, !!! ??? ... --- ;;; ,,, !!!"),
            "stop.pdf": pdf("the and of to in is it that the and of to in is it that the and of to in is")}
    r = check(build_client(), docs)
    assert r.status_code == 200, r.text
    d = r.json()
    status = {x["name"]: x["status"] for x in d["documents"]}
    assert status["a.pdf"] == "compared" and status["stop.pdf"] == "insufficient_text"
    assert d["pairs"] == [] and any("Fewer than two" in w for w in d["warnings"])


def test_repeated_single_word_is_insufficient_vocabulary(build_client):
    docs = {"a.pdf": pdf("banana " * 200), "b.pdf": pdf("banana " * 150), "c.pdf": pdf(PHOTO)}
    d = check(build_client(), docs).json()
    st = {x["name"]: x["status"] for x in d["documents"]}
    assert st["a.pdf"] == st["b.pdf"] == "insufficient_text" and d["pairs"] == []  # no bogus 100% match


def test_short_documents_warn(build_client):
    d = check(build_client(), {"a.pdf": pdf("alpha beta gamma delta epsilon zeta eta theta"),
                               "b.pdf": pdf("alpha beta gamma delta epsilon zeta eta theta iota")}).json()
    assert any("very short" in w for w in d["warnings"])


def test_one_corrupt_file_skipped_others_compared(build_client):
    d = check(build_client(), {"a.pdf": pdf(PHOTO), "b.pdf": pdf(PHOTO_COPY), "bad.pdf": b"%PDF-1.4 garbage"}).json()
    assert [s["name"] for s in d["skipped_documents"]] == ["bad.pdf"] and len(d["pairs"]) == 1


def test_all_files_unreadable_is_422_with_details(build_client):
    r = check(build_client(), {"a.pdf": b"%PDF-garbage", "b.pdf": b"%PDF-garbage2"})
    assert r.status_code == 422 and len(r.json()["error"]["details"]) == 2


def test_request_validation_and_upload_limits(build_client):
    c = build_client(plagiarism_max_files=3, max_upload_mb=1)
    ok = pdf(PHOTO)
    assert check(c, {"a.pdf": ok}).status_code == 400                                   # need >= 2
    assert check(c, {f"{i}.pdf": ok for i in range(4)}).status_code == 400              # too many
    assert check(c, {"a.pdf": ok, "b.exe": ok}).status_code == 415                      # bad extension
    assert check(c, {"a.pdf": ok, "b.pdf": b"MZ" + b"0" * 100}).status_code == 415      # content mismatch
    assert check(c, {"a.pdf": ok, "b.pdf": b"%PDF-" + b"0" * (1024 * 1024 + 5)}).status_code == 413
    assert c.post("/api/plagiarism/check").status_code == 422


def test_duplicate_filenames_disambiguated(build_client):
    r = build_client().post("/api/plagiarism/check", files=[("files", ("s.pdf", pdf(PHOTO))), ("files", ("s.pdf", pdf(PHOTO_COPY)))])
    names = [x["name"] for x in r.json()["documents"]]
    assert names == ["s.pdf", "s.pdf (2)"]


def test_path_traversal_filename_is_sanitised(build_client):
    r = build_client().post("/api/plagiarism/check", files=[("files", ("../../etc/a.pdf", pdf(PHOTO))), ("files", ("b.pdf", pdf(PHOTO_COPY)))])
    assert r.status_code == 200 and "etc" not in r.text and "/" not in r.json()["documents"][0]["name"]


# ---- images / OCR ------------------------------------------------------------------------
def test_image_uploads_use_ocr_and_compare(build_client):
    s = make_settings()
    c = build_client(settings=s, ocr=FakeOCR(s, text=PHOTO))
    img = make_text_image("x")
    r = check(c, {"a.png": img, "b.png": img})
    d = r.json()
    assert r.status_code == 200 and all(x["used_ocr"] for x in d["documents"]) and d["pairs"][0]["flagged_for_review"]
    assert any("OCR" in w for w in d["warnings"])


def test_ocr_unavailable_degrades_gracefully(build_client):
    r = check(build_client(), {"a.pdf": pdf(PHOTO), "b.pdf": pdf(PHOTO_COPY), "scan.png": make_text_image("hi")})
    d = r.json()
    assert r.status_code == 200 and d["skipped_documents"][0]["name"] == "scan.png" and len(d["pairs"]) == 1


def test_corrupt_image_skipped(build_client):
    s = make_settings()
    c = build_client(settings=s, ocr=FakeOCR(s, text=PHOTO))
    d = check(c, {"a.pdf": pdf(PHOTO), "b.pdf": pdf(PHOTO_COPY), "bad.png": b"\x89PNG\r\n\x1a\nbroken"}).json()
    assert [x["name"] for x in d["skipped_documents"]] == ["bad.png"]


# ---- optional Gemini paraphrase assessment ---------------------------------------------------
PARA_OK = {"likely_paraphrase": True, "confidence": 0.7, "explanation": "Sentences restate the same points."}


def test_ai_assessment_included_when_requested(build_client):
    ai = FakeAI([PARA_OK])
    d = check(build_client(ai=ai), {"a.pdf": pdf(PHOTO), "b.pdf": pdf(PHOTO_PARAPHRASE)}, use_ai=True).json()
    pa = d["pairs"][0]["paraphrase_assessment"]
    assert pa["status"] == "completed" and pa["likely_paraphrase"] is True and pa["limitations"] and "pair" in pa["note"]
    assert ai.calls and "<document>" in ai.calls[0]["prompt"]


def test_ai_not_called_by_default(build_client):
    ai = FakeAI([PARA_OK])
    d = check(build_client(ai=ai), {"a.pdf": pdf(PHOTO), "b.pdf": pdf(PHOTO_PARAPHRASE)}).json()
    assert ai.calls == [] and d["pairs"][0]["paraphrase_assessment"] is None


def test_ai_failure_does_not_fail_request(build_client):
    d = check(build_client(ai=FakeAI([AIUpstreamError()])), {"a.pdf": pdf(PHOTO), "b.pdf": pdf(PHOTO_PARAPHRASE)}, use_ai=True)
    assert d.status_code == 200
    pa = d.json()["pairs"][0]["paraphrase_assessment"]
    assert pa["status"] == "unavailable" and d.json()["pairs"][0]["document_similarity"] > 0


def test_ai_unconfigured_degrades(build_client):
    d = check(build_client(ai=FakeAI([PARA_OK], configured=False)), {"a.pdf": pdf(PHOTO), "b.pdf": pdf(PHOTO_PARAPHRASE)}, use_ai=True).json()
    assert d["pairs"][0]["paraphrase_assessment"]["status"] == "unavailable"


def test_ai_pair_limit_respected(build_client):
    ai = FakeAI([PARA_OK])
    docs = {f"{i}.pdf": pdf(PHOTO if i % 2 == 0 else PHOTO_PARAPHRASE) for i in range(4)}
    d = check(build_client(ai=ai, plagiarism_ai_max_pairs=1), docs, use_ai=True).json()
    assert len(ai.calls) <= 1 and any("limited" in w for w in d["warnings"])


def test_no_candidates_skips_ai(build_client):
    ai = FakeAI([PARA_OK])
    d = check(build_client(ai=ai), {"a.pdf": pdf(PHOTO), "b.pdf": pdf(PHOTO)}, use_ai=True).json()  # identical => nothing "moderate"
    assert ai.calls == [] and d["pairs"][0]["paraphrase_assessment"]["status"] == "skipped"


def test_true_paraphrase_is_low_tfidf_but_still_reaches_ai(build_client):
    """Regression: paraphrases score low on TF-IDF and must still be eligible for the AI opinion."""
    ai = FakeAI([PARA_OK])
    p = check(build_client(ai=ai), {"a.pdf": pdf(PHOTO), "b.pdf": pdf(PHOTO_PARAPHRASE)}, use_ai=True).json()["pairs"][0]
    assert p["document_similarity"] < 0.35 and p["flagged_for_review"] is False
    assert p["paraphrase_assessment"]["status"] == "completed" and len(ai.calls) == 1
