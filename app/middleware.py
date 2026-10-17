"""Small, dependency-free middleware: request-size guard and an in-memory rate limiter."""

from __future__ import annotations

import time
from collections import defaultdict, deque

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse


def _err(code: str, message: str, status: int, headers: dict | None = None) -> JSONResponse:
    return JSONResponse({"error": {"code": code, "message": message}}, status_code=status, headers=headers)


class RequestSizeLimitMiddleware(BaseHTTPMiddleware):
    """Reject requests whose declared Content-Length exceeds the limit, before the body is parsed.

    Note: a client that omits/lies about Content-Length is still bounded per-file by the
    upload reader; put a reverse proxy (nginx etc.) with a body limit in front in production.
    """

    def __init__(self, app, max_bytes: int) -> None:
        super().__init__(app)
        self._max = max_bytes

    async def dispatch(self, request: Request, call_next):
        length = request.headers.get("content-length")
        if length and length.isdigit() and int(length) > self._max:
            return _err("request_too_large", "Request body is too large.", 413)
        return await call_next(request)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Sliding-window limiter per client IP for /api/* paths.

    LIMITATIONS: state is per-process memory (resets on restart, not shared across workers),
    and behind a reverse proxy every client shares the proxy's IP unless the proxy is trusted
    and configured. This is a basic safeguard for a single-process deployment, not
    production-grade rate limiting.
    """

    def __init__(self, app, per_minute: int) -> None:
        super().__init__(app)
        self._limit = per_minute
        self._hits: dict[str, deque[float]] = defaultdict(deque)

    async def dispatch(self, request: Request, call_next):
        if self._limit <= 0 or not request.url.path.startswith("/api/"):
            return await call_next(request)
        client = request.client.host if request.client else "unknown"
        now = time.monotonic()
        window = self._hits[client]
        while window and now - window[0] > 60:
            window.popleft()
        if len(window) >= self._limit:
            retry = max(1, int(60 - (now - window[0])))
            return _err("rate_limited", "Too many requests. Please slow down.", 429, {"Retry-After": str(retry)})
        window.append(now)
        return await call_next(request)
