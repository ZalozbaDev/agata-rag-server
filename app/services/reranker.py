from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """Optional cross-encoder rerank stage.

    Hybrid fusion decides ordering from two weak signals; a cross-encoder reads
    the question and the passage together and is the only stage that can tell
    "lexically similar" from "actually answers the question". The model is
    loaded lazily so the dependency costs nothing while RERANK_ENABLED is off.
    """

    def __init__(self, model_name: str) -> None:
        self._model_name = model_name
        self._model: object | None = None
        self._lock = asyncio.Lock()

    def _load(self) -> object:
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        logger.info('Loading reranker model %s', self._model_name)
        return TextCrossEncoder(model_name=self._model_name)

    async def _ensure_model(self) -> object:
        if self._model is None:
            async with self._lock:
                if self._model is None:
                    self._model = await asyncio.to_thread(self._load)
        return self._model

    async def score(self, query: str, documents: Sequence[str]) -> list[float]:
        if not documents:
            return []

        model = await self._ensure_model()

        def _run() -> list[float]:
            return [float(value) for value in model.rerank(query, list(documents))]

        return await asyncio.to_thread(_run)

    async def warm_up(self) -> None:
        await self._ensure_model()
