import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from tests.helpers import FakeAI, FakeEmbedder, FakeOCR, make_settings


@pytest.fixture
def settings():
    return make_settings()


@pytest.fixture
def build_client():
    """Factory: build_client(ai=..., settings=..., ocr=...) -> TestClient with fakes injected."""

    def _build(ai=None, settings=None, ocr=None, embedder=None, **settings_kw) -> TestClient:
        s = settings or make_settings(**settings_kw)
        app = create_app(s, ai=ai if ai is not None else FakeAI([{}]), embedder=embedder or FakeEmbedder(),
                         ocr=ocr or FakeOCR(s, available=False))
        return TestClient(app, raise_server_exceptions=False)

    return _build
