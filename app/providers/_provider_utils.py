from __future__ import annotations

import httpx
from openai import APIConnectionError, APITimeoutError, APIStatusError, AsyncOpenAI, RateLimitError

from app.core.config import Settings
from app.core.errors import OpenAIServiceError, SotraServiceError
from app.utils.retry import retry_async


def _is_transient_openai_error(exc: Exception) -> bool:
    if isinstance(exc, (APITimeoutError, APIConnectionError, RateLimitError)):
        return True
    if isinstance(exc, APIStatusError):
        return exc.status_code >= 500 or exc.status_code == 429
    return False


def _is_transient_http_error(exc: Exception) -> bool:
    if isinstance(exc, httpx.TimeoutException):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code >= 500 or exc.response.status_code == 429
    if isinstance(exc, httpx.TransportError):
        return True
    return False
