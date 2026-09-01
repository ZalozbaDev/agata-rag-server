from __future__ import annotations

import asyncio

import pytest

from app.clients.qdrant_client import UnknownCollectionError
from app.core.config import Settings
from app.services.rag_service import RagService
from app.services.retrieval_service import RetrievalService


def make_settings(**overrides: object) -> Settings:
    defaults: dict[str, object] = {
        'app_name': 'test',
        'environment': 'local',
        'log_level': 'INFO',
        'api_prefix': '',
        'qdrant_url': 'http://localhost:6333',
        'qdrant_api_key': None,
        'qdrant_collection': 'documents',
        'qdrant_allowed_collections': [],
        'qdrant_timeout_seconds': 5.0,
        'document_language': 'hsb',
        'embedding_dimension': 8,
        'vector_distance': 'cosine',
        'hybrid_prefetch_limit': 20,
        'embedding_provider': 'openai',
        'llm_provider': 'openai',
        'openai_api_key': None,
        'openai_base_url': None,
        'openai_embedding_model': 'm',
        'openai_chat_model': 'm',
        'openai_timeout_seconds': 5.0,
        'ollama_url': None,
        'ollama_api_key': None,
        'ollama_embedding_model': 'm',
        'ollama_chat_model': 'm',
        'ollama_timeout_seconds': 5.0,
        'retrieval_top_k': 6,
        'retrieval_min_score': 0.0,
        'min_rag_hits': 1,
        'provider_max_retries': 1,
        'provider_retry_base_delay_seconds': 0.0,
        'chunk_size': 100,
        'chunk_overlap': 10,
        'max_context_chunks': 8,
        'scheduler_enabled': False,
        'scheduler_interval_hours': 24,
        'reparse_urls': [],
        'cors_origins': ['*'],
        'cors_allow_credentials': False,
    }
    defaults.update(overrides)
    return Settings(**defaults)  # type: ignore[arg-type]


class FakeGateway:
    """Records what the services ask Qdrant for."""

    def __init__(self, settings: Settings, hits: list[object] | None = None) -> None:
        self.settings = settings
        self.hits = hits or []
        self.ensured: list[str] = []
        self.searched: list[dict[str, object]] = []

    def resolve_collection(self, collection: str | None) -> str:
        from app.clients.qdrant_client import QdrantGateway

        return QdrantGateway.resolve_collection(self, collection)  # type: ignore[arg-type]

    async def ensure_collection(self, collection: str | None = None) -> None:
        self.ensured.append(self.resolve_collection(collection))

    async def hybrid_search(self, **kwargs: object) -> list[object]:
        self.searched.append(kwargs)
        return self.hits


class FakeEmbeddings:
    async def embed_query(self, _text: str) -> list[float]:
        return [0.0]

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] for _ in texts]


class FakeSparse:
    async def embed_query_async(self, _text: str) -> object:
        return object()

    async def embed_texts_async(self, texts: list[str]) -> list[object]:
        return [object() for _ in texts]


def make_retrieval(gateway: FakeGateway) -> RetrievalService:
    return RetrievalService(
        qdrant=gateway,  # type: ignore[arg-type]
        embeddings=FakeEmbeddings(),  # type: ignore[arg-type]
        sparse_embeddings=FakeSparse(),  # type: ignore[arg-type]
        hybrid_prefetch_limit=20,
    )


def test_resolve_collection_defaults_to_settings() -> None:
    gateway = FakeGateway(make_settings())
    assert gateway.resolve_collection(None) == 'documents'
    assert gateway.resolve_collection('  ') == 'documents'


def test_resolve_collection_rejects_unsafe_names() -> None:
    gateway = FakeGateway(make_settings())
    for name in ('../evil', 'with space', 'a' * 80, 'semi;colon'):
        with pytest.raises(UnknownCollectionError):
            gateway.resolve_collection(name)


def test_resolve_collection_honours_allowlist() -> None:
    gateway = FakeGateway(
        make_settings(qdrant_allowed_collections=['zalozba_strukturna_zmena'])
    )
    assert gateway.resolve_collection('zalozba_strukturna_zmena') == 'zalozba_strukturna_zmena'
    # The default stays reachable even when it is not listed explicitly.
    assert gateway.resolve_collection('documents') == 'documents'
    with pytest.raises(UnknownCollectionError):
        gateway.resolve_collection('some_other_collection')


def test_retrieval_targets_requested_collection() -> None:
    gateway = FakeGateway(make_settings())
    asyncio.run(
        make_retrieval(gateway).retrieve('question', 4, 'zalozba_spechowane_projekty')
    )

    assert gateway.ensured == ['zalozba_spechowane_projekty']
    assert gateway.searched[0]['collection'] == 'zalozba_spechowane_projekty'
    assert gateway.searched[0]['limit'] == 4


def test_retrieval_falls_back_to_default_collection() -> None:
    gateway = FakeGateway(make_settings())
    asyncio.run(make_retrieval(gateway).retrieve('question', 6, None))

    assert gateway.searched[0]['collection'] == 'documents'


def test_ask_top_k_overrides_configured_default() -> None:
    gateway = FakeGateway(make_settings())
    service = RagService(
        retrieval=make_retrieval(gateway),
        top_k=6,
        max_context_chunks=8,
        retrieval_min_score=0.0,
        min_rag_hits=1,
    )

    asyncio.run(service.retrieve('question', collection='documents', top_k=2))
    assert gateway.searched[0]['limit'] == 2

    asyncio.run(service.retrieve('question', collection='documents'))
    assert gateway.searched[1]['limit'] == 6
