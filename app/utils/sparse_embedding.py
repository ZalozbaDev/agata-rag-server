from __future__ import annotations

import asyncio
from collections.abc import Sequence

from fastembed import SparseTextEmbedding
from qdrant_client import models


class SparseEmbeddingProvider:
    def __init__(self, model_name: str = 'Qdrant/bm25') -> None:
        self._model = SparseTextEmbedding(model_name=model_name)

    def _to_sparse_vector(self, embedding: object) -> models.SparseVector:
        indices = list(getattr(embedding, 'indices'))
        values = [float(value) for value in getattr(embedding, 'values')]
        return models.SparseVector(indices=indices, values=values)

    def embed_texts(self, texts: Sequence[str]) -> list[models.SparseVector]:
        normalized = [text.strip() for text in texts]
        if not normalized:
            return []

        embeddings = list(self._model.embed(normalized))
        return [self._to_sparse_vector(item) for item in embeddings]

    def embed_query(self, text: str) -> models.SparseVector:
        vectors = self.embed_texts([text])
        return vectors[0]

    async def embed_texts_async(self, texts: Sequence[str]) -> list[models.SparseVector]:
        return await asyncio.to_thread(self.embed_texts, texts)

    async def embed_query_async(self, text: str) -> models.SparseVector:
        return await asyncio.to_thread(self.embed_query, text)
