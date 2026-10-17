"""Health, error format, config, rate limit, size guard."""

from app.config import Settings
from tests.helpers import FakeAI


def test_health_ok_and_does_not_need_key(build_client):
    c = build_client(settings=Settings(_env_file=None, rate_limit_per_minute=0))  # no key at all
    r = c.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok" and body["gemini_configured"] is False
    assert set(body) == {"status", "version", "gemini_configured", "ocr_available"}


def test_openapi_docs_load_and_list_required_endpoints(build_client):
    c = build_client()
    assert c.get("/docs").status_code == 200
    paths = c.get("/openapi.json").json()["paths"]
    for p in ["/health", "/api/rag/ask", "/api/questions/generate", "/api/plagiarism/check",
              "/api/evaluation/evaluate", "/api/evaluation/evaluate-batch"]:
        assert p in paths


def test_missing_api_key_returns_503_on_ai_endpoints(build_client):
    from app.services.gemini_service import GeminiService
    s = Settings(_env_file=None, rate_limit_per_minute=0)
    from app.main import create_app
    from fastapi.testclient import TestClient
    c = TestClient(create_app(s, ai=GeminiService(s)), raise_server_exceptions=False)
    q = c.post("/api/questions/generate", json={"source_text": "x" * 80, "num_questions": 1, "difficulty": "easy"})
    e = c.post("/api/evaluation/evaluate", json={
        "question": "q", "reference_answer": "a", "rubric_text": "r", "max_marks": 2, "student_answer": "some answer"})
    for r in (q, e):
        assert r.status_code == 503
        assert r.json()["error"]["code"] == "ai_not_configured"


def test_validation_error_format_does_not_echo_input(build_client):
    c = build_client()
    secret = "STUDENT-SECRET-ANSWER"
    r = c.post("/api/evaluation/evaluate", json={"student_answer": secret, "max_marks": "abc"})
    assert r.status_code == 422
    err = r.json()["error"]
    assert err["code"] == "validation_error" and err["details"]
    assert secret not in r.text


def test_unknown_route_uses_error_format(build_client):
    r = build_client().get("/nope")
    assert r.status_code == 404 and r.json()["error"]["code"] == "http_error"


def test_json_endpoints_reject_form_data(build_client):
    c = build_client()
    r = c.post("/api/questions/generate", data={"source_text": "x"})
    assert r.status_code == 422


def test_unhandled_exception_is_sanitised(build_client):
    class Boom(FakeAI):
        async def generate_json(self, **kw):
            raise RuntimeError("secret-key-abc leaked?")

    c = build_client(ai=Boom())
    r = c.post("/api/evaluation/evaluate", json={
        "question": "q", "reference_answer": "a", "rubric_text": "r", "max_marks": 2, "student_answer": "x"})
    assert r.status_code == 500
    assert "secret-key-abc" not in r.text and r.json()["error"]["code"] == "internal_error"


def test_rate_limit_returns_429_with_retry_after(build_client):
    c = build_client(rate_limit_per_minute=2)
    body = {"source_text": "x" * 80, "num_questions": 0, "difficulty": "easy"}
    assert c.post("/api/questions/generate", json=body).status_code == 422
    assert c.post("/api/questions/generate", json=body).status_code == 422
    r = c.post("/api/questions/generate", json=body)
    assert r.status_code == 429 and "retry-after" in r.headers
    assert c.get("/health").status_code == 200  # health is not limited


def test_oversized_request_rejected_before_parsing(build_client):
    c = build_client(max_upload_mb=1, plagiarism_max_files=2)
    r = c.post("/api/plagiarism/check", files=[("files", ("a.pdf", b"%PDF-" + b"0" * (4 * 1024 * 1024)))])
    assert r.status_code == 413
