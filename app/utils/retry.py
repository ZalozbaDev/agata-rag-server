from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import TypeVar

T = TypeVar('T')


async def retry_async(
    fn: Callable[[], Awaitable[T]],
    *,
    max_attempts: int = 3,
    base_delay: float = 0.5,
    retryable: Callable[[Exception], bool] | None = None,
) -> T:
    if max_attempts < 1:
        raise ValueError('max_attempts must be >= 1')

    last_exc: Exception | None = None

    for attempt in range(max_attempts):
        try:
            return await fn()
        except Exception as exc:
            last_exc = exc
            if attempt >= max_attempts - 1:
                break
            if retryable is not None and not retryable(exc):
                raise
            await asyncio.sleep(base_delay * (2**attempt))

    assert last_exc is not None
    raise last_exc
