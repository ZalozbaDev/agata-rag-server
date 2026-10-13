from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence
from dataclasses import dataclass

from fastembed import SparseTextEmbedding
from qdrant_client import models

logger = logging.getLogger(__name__)

# Snowball languages shipped with Qdrant/bm25. Czech and Polish are *not* on
# this list even though Snowball has them; Qdrant/bm25 only ships stopword
# files for the languages below. Russian is the only Slavic option.
SUPPORTED_BM25_LANGUAGES = frozenset(
    {
        'arabic',
        'danish',
        'dutch',
        'english',
        'finnish',
        'french',
        'german',
        'greek',
        'hungarian',
        'italian',
        'norwegian',
        'portuguese',
        'romanian',
        'russian',
        'spanish',
        'swedish',
        'tamil',
        'turkish',
    }
)

_LANGUAGE_ALIASES = {
    'de': 'german',
    'deu': 'german',
    'ger': 'german',
    'en': 'english',
    'eng': 'english',
}


@dataclass(frozen=True)
class Bm25Config:
    """Snowball settings for one document/query language."""

    snowball: str
    disable_stemmer: bool
    key: str


def resolve_bm25_config(
    language: str | None,
    default: str = 'german',
) -> Bm25Config:
    """Pick BM25 stemming/stopwords for a payload language.

    Upper Sorbian has no Snowball stemmer, and Qdrant/bm25 does not ship Czech
    or Polish. German/English stopword lists would eat real Sorbian function
    words (``so``, ``a``). ``disable_stemmer`` also clears stopwords in this
    library, which is what we want: keep surface forms, drop no tokens. The
    Snowball name is only a constructor requirement in that case.
    """
    raw = (language or '').strip().lower()
    if raw in {'hsb', 'dsb'}:
        return Bm25Config(snowball='german', disable_stemmer=True, key='hsb-raw')

    if not raw:
        snowball = default if default in SUPPORTED_BM25_LANGUAGES else 'german'
    else:
        snowball = _LANGUAGE_ALIASES.get(raw, raw)
        if snowball not in SUPPORTED_BM25_LANGUAGES:
            logger.warning(
                'BM25 language %r is not supported; falling back to %r',
                language,
                default,
            )
            snowball = default if default in SUPPORTED_BM25_LANGUAGES else 'german'

    return Bm25Config(snowball=snowball, disable_stemmer=False, key=snowball)


def normalize_bm25_language(language: str | None, default: str = 'german') -> str:
    """Snowball language name used for this request (tests and logs)."""
    return resolve_bm25_config(language, default).snowball


class SparseEmbeddingProvider:
    """BM25 sparse vectors, with one model instance per stemming language.

    Documents and queries deliberately use different fastembed entry points:
    `embed` applies BM25 term-frequency weighting (correct for passages), while
    `query_embed` emits plain term indicators, which is what Qdrant's IDF
    modifier expects on the query side.
    """

    def __init__(
        self,
        model_name: str = 'Qdrant/bm25',
        default_language: str = 'german',
    ) -> None:
        self._model_name = model_name
        self._default_language = normalize_bm25_language(default_language)
        self._models: dict[str, SparseTextEmbedding] = {}
        self._lock = asyncio.Lock()

    def _model_for(self, language: str | None) -> SparseTextEmbedding:
        config = resolve_bm25_config(language, self._default_language)
        model = self._models.get(config.key)
        if model is None:
            logger.info(
                'Loading BM25 model snowball=%s stemmer=%s key=%s',
                config.snowball,
                'off' if config.disable_stemmer else 'on',
                config.key,
            )
            model = SparseTextEmbedding(
                model_name=self._model_name,
                language=config.snowball,
                disable_stemmer=config.disable_stemmer,
            )
            self._models[config.key] = model
        return model

    @staticmethod
    def _to_sparse_vector(embedding: object) -> models.SparseVector:
        indices = [int(index) for index in getattr(embedding, 'indices')]
        values = [float(value) for value in getattr(embedding, 'values')]
        return models.SparseVector(indices=indices, values=values)

    def embed_texts(
        self,
        texts: Sequence[str],
        language: str | None = None,
    ) -> list[models.SparseVector]:
        normalized = [text.strip() for text in texts]
        if not normalized:
            return []

        embeddings = list(self._model_for(language).embed(normalized))
        return [self._to_sparse_vector(item) for item in embeddings]

    def embed_query(
        self,
        text: str,
        language: str | None = None,
    ) -> models.SparseVector:
        embeddings = list(self._model_for(language).query_embed(text.strip()))
        if not embeddings:
            return models.SparseVector(indices=[], values=[])
        return self._to_sparse_vector(embeddings[0])

    async def embed_texts_async(
        self,
        texts: Sequence[str],
        language: str | None = None,
    ) -> list[models.SparseVector]:
        return await asyncio.to_thread(self.embed_texts, texts, language)

    async def embed_query_async(
        self,
        text: str,
        language: str | None = None,
    ) -> models.SparseVector:
        return await asyncio.to_thread(self.embed_query, text, language)

    async def warm_up(self, languages: Sequence[str]) -> None:
        """Load models up front so the first user query is not slowed down."""
        async with self._lock:
            for language in languages:
                await asyncio.to_thread(self._model_for, language)
