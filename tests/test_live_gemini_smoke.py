"""OPTIONAL smoke test against the real Gemini API. Skipped unless explicitly enabled.

    RUN_LIVE_GEMINI=1 GEMINI_API_KEY=... pytest tests/test_live_gemini_smoke.py -v

This costs a (tiny) amount of API quota and needs network access.
"""

import os

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app

pytestmark = pytest.mark.skipif(
    os.getenv("RUN_LIVE_GEMINI") != "1" or not os.getenv("GEMINI_API_KEY"),
    reason="set RUN_LIVE_GEMINI=1 and GEMINI_API_KEY to run live smoke tests",
)


def test_live_evaluation_and_questions():
    client = TestClient(create_app(Settings(_env_file=None, rate_limit_per_minute=0)))
    r = client.post("/api/evaluation/evaluate", json={
        "question": "What is photosynthesis?", "reference_answer": "Plants convert light, water and CO2 into glucose and oxygen.",
        "rubric": [{"description": "Mentions light energy", "max_marks": 1}, {"description": "Mentions glucose/oxygen", "max_marks": 1}],
        "max_marks": 2, "student_answer": "Plants use sunlight to make sugar and give off oxygen."})
    assert r.status_code == 200, r.text
    assert 0 <= r.json()["awarded_marks"] <= 2
    q = client.post("/api/questions/generate", json={
        "source_text": "Photosynthesis is the process by which green plants convert light energy into chemical energy. "
                       "It occurs in chloroplasts and produces glucose and oxygen from carbon dioxide and water.",
        "num_questions": 2, "difficulty": "easy", "question_types": ["mcq", "short_answer"]})
    assert q.status_code == 200, q.text
    assert len(q.json()["questions"]) == 2
