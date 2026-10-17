"""GeminiService behaviour against a fake SDK client: retries, error mapping, JSON parsing."""

import asyncio

import httpx
import pytest
from google.genai import errors as genai_errors
from pydantic import BaseModel

from app.errors import (AIAuthenticationError, AIConfigurationError, AIInvalidOutputError, AIRateLimitedError,
                        AITimeoutError, AIUpstreamError, AIModelUnavailableError)
from app.services.gemini_service import GeminiService
from tests.helpers import make_settings


class Out(BaseModel):
    answer: str
    score: int


class Resp:
    def __init__(self, text, feedback=None):
        self.text, self.prompt_feedback = text, feedback


class FakeClient:
    def __init__(self, script):
        self.script, self.calls = list(script), 0
        outer = self

        class Models:
            async def generate_content(self, **kw):
                outer.calls += 1
                item = outer.script.pop(0)
                if isinstance(item, Exception):
                    raise item
                return item

        class Aio:
            models = Models()

        self.aio = Aio()


def svc(script, **kw):
    sleeps = []

    async def fake_sleep(d):
        sleeps.append(d)

    client = FakeClient(script)
    return GeminiService(make_settings(**kw), client=client, sleep=fake_sleep), client, sleeps


def api_err(code):
    cls = genai_errors.ServerError if code >= 500 else genai_errors.ClientError
    return cls(code, {"error": {"message": "provider detail with key=SECRET123"}})


async def call(s):
    return await s.generate_json(prompt="p", schema=Out, system_instruction="sys")


def test_success_and_code_fence_stripped():
    s, c, _ = svc([Resp('```json\n{"answer": "a", "score": 3}\n```')])
    out = asyncio.run(call(s))
    assert out == Out(answer="a", score=3) and c.calls == 1


def test_retries_transient_then_succeeds_with_bounded_backoff():
    s, c, sleeps = svc([api_err(503), api_err(429), Resp('{"answer":"a","score":1}')], gemini_max_retries=2,
                       gemini_backoff_base_seconds=1.0)
    assert asyncio.run(call(s)).score == 1
    assert c.calls == 3 and len(sleeps) == 2 and sleeps[1] > sleeps[0]


def test_retries_are_bounded_then_maps_error():
    s, c, _ = svc([api_err(503)] * 10, gemini_max_retries=2)
    with pytest.raises(AIUpstreamError):
        asyncio.run(call(s))
    assert c.calls == 3  # 1 + 2 retries, never unlimited


@pytest.mark.parametrize("code,exc_type", [(401, AIAuthenticationError), (403, AIAuthenticationError),
                                           (404, AIModelUnavailableError)])
def test_non_retryable_errors_fail_immediately(code, exc_type):
    s, c, _ = svc([api_err(code), Resp('{"answer":"a","score":1}')])
    with pytest.raises(exc_type):
        asyncio.run(call(s))
    assert c.calls == 1


def test_400_is_not_retried():
    s, c, _ = svc([api_err(400)])
    with pytest.raises(AIUpstreamError):
        asyncio.run(call(s))
    assert c.calls == 1


def test_rate_limit_exhausted_maps_to_429():
    s, c, _ = svc([api_err(429)] * 5, gemini_max_retries=1)
    with pytest.raises(AIRateLimitedError):
        asyncio.run(call(s))
    assert c.calls == 2


def test_timeout_is_retried_then_mapped():
    s, c, _ = svc([httpx.ReadTimeout("slow")] * 5, gemini_max_retries=1)
    with pytest.raises(AITimeoutError):
        asyncio.run(call(s))
    assert c.calls == 2


def test_error_messages_never_leak_provider_detail_or_key():
    s, _, _ = svc([api_err(401)])
    with pytest.raises(AIAuthenticationError) as ei:
        asyncio.run(call(s))
    assert "SECRET123" not in str(ei.value) and "SECRET123" not in ei.value.message


def test_malformed_json_reasked_once_then_succeeds():
    s, c, _ = svc([Resp("{not json"), Resp('{"answer":"ok","score":2}')], gemini_invalid_output_retries=1)
    assert asyncio.run(call(s)).answer == "ok" and c.calls == 2


def test_malformed_json_raises_after_budget():
    s, c, _ = svc([Resp("{not json"), Resp('{"answer": 5}')], gemini_invalid_output_retries=1)
    with pytest.raises(AIInvalidOutputError):
        asyncio.run(call(s))
    assert c.calls == 2


def test_schema_violation_rejected():
    s, _, _ = svc([Resp('{"answer":"a","score":"high"}')], gemini_invalid_output_retries=0)
    with pytest.raises(AIInvalidOutputError):
        asyncio.run(call(s))


def test_empty_or_blocked_response_is_invalid_output():
    s, _, _ = svc([Resp(None)], gemini_invalid_output_retries=0)
    with pytest.raises(AIInvalidOutputError):
        asyncio.run(call(s))


def test_missing_key_raises_configuration_error_without_network():
    from app.config import Settings
    s = GeminiService(Settings(_env_file=None))
    assert s.is_configured is False
    with pytest.raises(AIConfigurationError):
        asyncio.run(call(s))


def test_api_key_not_in_settings_repr():
    st = make_settings(gemini_api_key="super-secret-value")
    assert "super-secret-value" not in repr(st) and "super-secret-value" not in str(st)
