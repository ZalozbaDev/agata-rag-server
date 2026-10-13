from __future__ import annotations

import json
import os
from functools import lru_cache

from pathlib import Path
from typing import Literal, cast

from pydantic import BaseModel


def _load_dotenv(path: str = '.env') -> None:
    env_path = Path(path)
    if not env_path.is_file():
        return
    parsed: dict[str, str] = {}
    for raw_line in env_path.read_text(encoding='utf-8').splitlines():
        line = raw_line.strip()
        if not line or line.startswith('#') or '=' not in line:
            continue
        key, _, value = line.partition('=')
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if key:
            parsed[key] = value
    for key, value in parsed.items():
        if key not in os.environ:
            os.environ[key] = value


_load_dotenv()


def _env_bool(key: str, default: bool) -> bool:
    raw = os.getenv(key)
    if raw is None or raw.strip() == '':
        return default
    return raw.strip().lower() in {'1', 'true', 'yes', 'on'}


def _parse_list(value: str | None) -> list[str]:
    if value is None:
        return []
    cleaned = value.strip()
    if not cleaned:
        return []
    if cleaned.startswith('['):
        loaded = json.loads(cleaned)
        return [str(v).strip() for v in loaded if str(v).strip()]
    return [part.strip() for part in cleaned.split(',') if part.strip()]


def _parse_cors_origins(value: str | None) -> list[str]:
    parsed = _parse_list(value)
    return parsed or ['*']


class Settings(BaseModel):
    app_name: str
    environment: Literal['local', 'dev', 'prod']
    log_level: str
    api_prefix: str

    qdrant_url: str
    qdrant_api_key: str | None
    qdrant_collection: str
    qdrant_allowed_collections: list[str]
    qdrant_timeout_seconds: float
    document_language: str
    embedding_dimension: int
    vector_distance: Literal['cosine', 'dot', 'euclid']
    hybrid_prefetch_limit: int

    embedding_provider: Literal['openai', 'ollama']
    embedding_prompt_style: str
    embedding_batch_size: int
    embedding_max_parallel_batches: int
    query_embedding_cache_size: int
    llm_provider: Literal['openai', 'ollama']
    openai_api_key: str | None
    openai_base_url: str | None
    openai_embedding_model: str
    openai_chat_model: str
    openai_timeout_seconds: float

    ollama_url: str | None
    ollama_api_key: str | None
    ollama_embedding_model: str
    ollama_chat_model: str
    ollama_timeout_seconds: float

    retrieval_top_k: int
    retrieval_score_threshold: float
    retrieval_candidate_multiplier: int
    min_rag_hits: int

    bm25_language: str

    rerank_enabled: bool
    rerank_model: str
    rerank_score_threshold: float

    dedup_similarity_threshold: float
    merge_adjacent_chunks: bool

    upsert_batch_size: int

    provider_max_retries: int
    provider_retry_base_delay_seconds: float

    chunk_size: int
    chunk_overlap: int
    max_context_chunks: int

    scheduler_enabled: bool
    scheduler_interval_hours: int
    reparse_urls: list[str]

    cors_origins: list[str]
    cors_allow_credentials: bool


@lru_cache
def get_settings() -> Settings:
    return Settings(
        app_name=os.getenv('APP_NAME', 'rag-server'),
        environment=cast(Literal['local', 'dev', 'prod'], os.getenv('ENVIRONMENT', 'local')),
        log_level=os.getenv('LOG_LEVEL', 'INFO'),
        api_prefix=os.getenv('API_PREFIX', ''),
        qdrant_url=os.getenv('QDRANT_URL', 'http://qdrant:6333'),
        qdrant_api_key=os.getenv('QDRANT_API_KEY') or None,
        qdrant_collection=os.getenv('QDRANT_COLLECTION', 'documents'),
        qdrant_allowed_collections=_parse_list(os.getenv('QDRANT_ALLOWED_COLLECTIONS')),
        qdrant_timeout_seconds=float(os.getenv('QDRANT_TIMEOUT_SECONDS', '20')),
        # Only used when a document carries no de/hsb signal at all; the
        # indexer detects the language from the text otherwise.
        document_language=os.getenv('DOCUMENT_LANGUAGE', 'de'),
        embedding_dimension=int(os.getenv('EMBEDDING_DIMENSION', '768')),
        vector_distance=cast(
            Literal['cosine', 'dot', 'euclid'],
            os.getenv('VECTOR_DISTANCE', 'cosine'),
        ),
        hybrid_prefetch_limit=int(os.getenv('HYBRID_PREFETCH_LIMIT', '20')),
        embedding_provider=cast(
            Literal['openai', 'ollama'],
            os.getenv('LLM_EMBEDDING_BACKEND', 'openai'),
        ),
        # 'gemma' | 'none' | '' (auto-detect from the model name)
        embedding_prompt_style=os.getenv('EMBEDDING_PROMPT_STYLE', ''),
        embedding_batch_size=int(os.getenv('EMBEDDING_BATCH_SIZE', '32')),
        embedding_max_parallel_batches=int(
            os.getenv('EMBEDDING_MAX_PARALLEL_BATCHES', '4')
        ),
        query_embedding_cache_size=int(os.getenv('QUERY_EMBEDDING_CACHE_SIZE', '512')),
        llm_provider=cast(
            Literal['openai', 'ollama'],
            os.getenv('LLM_EMBEDDING_BACKEND', 'openai'),
        ),
        openai_api_key=os.getenv('OPENAI_API_KEY') or None,
        openai_base_url=os.getenv('OPENAI_BASE_URL') or None,
        openai_embedding_model=os.getenv('OPENAI_EMBEDDING_MODEL', 'text-embedding-3-small'),
        openai_chat_model=os.getenv('OPENAI_CHAT_MODEL', 'gpt-5-mini'),
        openai_timeout_seconds=float(os.getenv('OPENAI_TIMEOUT_SECONDS', '60')),
        ollama_url=os.getenv('LLM_EMBEDDING_ENDPOINT') or None,
        ollama_api_key=os.getenv('LLM_EMBEDDING_API_KEY') or None,
        ollama_embedding_model=os.getenv('LLM_EMBEDDING_MODEL', 'embeddinggemma'),
        ollama_chat_model=os.getenv('LLM_EMBEDDING_MODEL', 'qwen3.5:4b'),
        ollama_timeout_seconds=float(os.getenv('OLLAMA_TIMEOUT_SECONDS', '60')),
        retrieval_top_k=int(os.getenv('RETRIEVAL_TOP_K', '6')),
        # Cosine similarity floor applied to the dense candidates. Replaces the
        # old RETRIEVAL_MIN_SCORE, which filtered RRF fusion scores and was
        # therefore a rank cutoff rather than a relevance gate.
        retrieval_score_threshold=float(os.getenv('RETRIEVAL_SCORE_THRESHOLD', '0.4')),
        retrieval_candidate_multiplier=int(
            os.getenv('RETRIEVAL_CANDIDATE_MULTIPLIER', '4')
        ),
        min_rag_hits=int(os.getenv('MIN_RAG_HITS', '1')),
        bm25_language=os.getenv('BM25_LANGUAGE', 'german'),
        rerank_enabled=_env_bool('RERANK_ENABLED', False),
        rerank_model=os.getenv(
            'RERANK_MODEL', 'jinaai/jina-reranker-v2-base-multilingual'
        ),
        rerank_score_threshold=float(os.getenv('RERANK_SCORE_THRESHOLD', '0.0')),
        dedup_similarity_threshold=float(
            os.getenv('DEDUP_SIMILARITY_THRESHOLD', '0.85')
        ),
        merge_adjacent_chunks=_env_bool('MERGE_ADJACENT_CHUNKS', True),
        upsert_batch_size=int(os.getenv('UPSERT_BATCH_SIZE', '128')),
        provider_max_retries=int(os.getenv('PROVIDER_MAX_RETRIES', '3')),
        provider_retry_base_delay_seconds=float(
            os.getenv('PROVIDER_RETRY_BASE_DELAY_SECONDS', '0.5')
        ),
        chunk_size=int(os.getenv('CHUNK_SIZE', '1000')),
        chunk_overlap=int(os.getenv('CHUNK_OVERLAP', '200')),
        max_context_chunks=int(os.getenv('MAX_CONTEXT_CHUNKS', '8')),
        scheduler_enabled=_env_bool('SCHEDULER_ENABLED', True),
        scheduler_interval_hours=int(os.getenv('SCHEDULER_INTERVAL_HOURS', '24')),
        reparse_urls=_parse_list(os.getenv('REPARSE_URLS')),
        cors_origins=_parse_cors_origins(os.getenv('CORS_ORIGINS')),
        cors_allow_credentials=_env_bool('CORS_ALLOW_CREDENTIALS', False),
    )
