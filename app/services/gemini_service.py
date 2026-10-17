"""Centralised Gemini client.

All AI calls in the app go through :meth:`GeminiService.generate_json`, which:

* reads the key/model from settings (never hardcoded, never logged),
* asks Gemini for schema-constrained JSON and re-validates it with Pydantic,
* retries only transient failures (429, 5xx, timeouts) a bounded number of times
  with exponential backoff + jitter,
* maps every failure to a sanitised :class:`AppError`.

Services depend on the small :class:`AIService` protocol, so tests substitute a fake.
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import re
from typing import Awaitable, Callable, Protocol, TypeVar

import httpx
from google import genai
from google.genai import errors as genai_errors
from google.genai import types
from pydantic import BaseModel, ValidationError

from app.config import Settings
from app.errors import (
    AIAuthenticationError,
    AIConfigurationError,
    AIInvalidOutputError,
    AIModelUnavailableError,
    AIRateLimitedError,
    AITimeoutError,
    AIUpstreamError,
)
from app.utils.prompts import UNTRUSTED_NOTICE

logger = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)

_FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.I)
_RETRYABLE_STATUS = {429, 500, 502, 503, 504}


class AIService(Protocol):
    @property
    def is_configured(self) -> bool: ...

    async def generate_json(
        self, *, prompt: str, schema: type[T], system_instruction: str, temperature: float = 0.2
    ) -> T: ...


class GeminiService:
    def __init__(
        self,
        settings: Settings,
        client: object | None = None,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self._settings = settings
        self._client = client
        self._sleep = sleep

    @property
    def is_configured(self) -> bool:
        return self._client is not None or self._settings.gemini_configured

    def _get_client(self):
        if self._client is None:
            if not self._settings.gemini_configured:
                raise AIConfigurationError()
            self._client = genai.Client(
                api_key=self._settings.gemini_api_key.get_secret_value(),  # type: ignore[union-attr]
                http_options=types.HttpOptions(timeout=int(self._settings.gemini_timeout_seconds * 1000)),
            )
        return self._client

    async def generate_json(
        self, *, prompt: str, schema: type[T], system_instruction: str, temperature: float = 0.2
    ) -> T:
        client = self._get_client()
        config = types.GenerateContentConfig(
            system_instruction=f"{system_instruction}\n\n{UNTRUSTED_NOTICE}",
            temperature=temperature,
            response_mime_type="application/json",
            response_schema=schema,
        )
        invalid_budget = self._settings.gemini_invalid_output_retries
        while True:
            raw = await self._call_with_retries(client, prompt, config)
            try:
                return self._parse(raw, schema)
            except AIInvalidOutputError:
                if invalid_budget <= 0:
                    raise
                invalid_budget -= 1
                logger.warning("Gemini returned invalid JSON; re-asking (%d left)", invalid_budget)

    # -- internals ---------------------------------------------------------
    async def _call_with_retries(self, client, prompt: str, config) -> str:
        max_attempts = self._settings.gemini_max_retries + 1
        for attempt in range(1, max_attempts + 1):
            try:
                response = await asyncio.wait_for(
                    client.aio.models.generate_content(
                        model=self._settings.gemini_model, contents=prompt, config=config
                    ),
                    timeout=self._settings.gemini_timeout_seconds + 5,
                )
                return self._extract_text(response)
            except AIInvalidOutputError:
                raise
            except Exception as exc:
                mapped, retryable = self._classify(exc)
                logger.warning(
                    "Gemini call failed (attempt %d/%d, model=%s): %s -> %s",
                    attempt, max_attempts, self._settings.gemini_model, type(exc).__name__, mapped.code,
                )
                if not retryable or attempt == max_attempts:
                    raise mapped from None
                delay = min(
                    self._settings.gemini_backoff_base_seconds * 2 ** (attempt - 1),
                    self._settings.gemini_backoff_max_seconds,
                )
                await self._sleep(delay + random.uniform(0, delay * 0.25))
        raise AIUpstreamError()  # pragma: no cover - loop always returns or raises

    @staticmethod
    def _extract_text(response) -> str:
        text = getattr(response, "text", None)
        if not text or not text.strip():
            feedback = getattr(response, "prompt_feedback", None)
            reason = getattr(feedback, "block_reason", None)
            raise AIInvalidOutputError(
                "The AI provider returned no content" + (" (blocked by safety filters)." if reason else ".")
            )
        return text

    @staticmethod
    def _parse(raw: str, schema: type[T]) -> T:
        cleaned = _FENCE.sub("", raw).strip()
        try:
            return schema.model_validate_json(cleaned)
        except (ValidationError, ValueError, json.JSONDecodeError) as exc:
            logger.warning("Gemini output failed %s validation (%d chars)", schema.__name__, len(raw))
            raise AIInvalidOutputError() from exc

    @staticmethod
    def _classify(exc: Exception) -> tuple[Exception, bool]:
        """Return (sanitised AppError, retryable?)."""
        if isinstance(exc, AIConfigurationError):
            return exc, False
        if isinstance(exc, (asyncio.TimeoutError, httpx.TimeoutException, TimeoutError)):
            return AITimeoutError(), True
        if isinstance(exc, genai_errors.APIError):
            code = int(getattr(exc, "code", 0) or 0)
            if code == 429:
                return AIRateLimitedError(), True
            if code in (401, 403):
                return AIAuthenticationError(), False
            if code == 404:
                return AIModelUnavailableError(), False
            if code in _RETRYABLE_STATUS:
                return (AITimeoutError() if code == 504 else AIUpstreamError()), True
            return AIUpstreamError("The AI provider rejected the request."), False
        if isinstance(exc, httpx.TransportError):
            return AIUpstreamError("Could not reach the AI provider."), True
        return AIUpstreamError(), False
