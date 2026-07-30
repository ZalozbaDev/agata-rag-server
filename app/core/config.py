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
    qdrant_timeout_seconds: float
    embedding_dimension: int
    vector_distance: Literal['cosine', 'dot', 'euclid']
    hybrid_prefetch_limit: int

    embedding_provider: Literal['openai', 'ollama']
    llm_provider: Literal['openai', 'ollama']
    openai_api_key: str | None
    openai_base_url: str | None
    openai_embedding_model: str
    openai_chat_model: str
    openai_timeout_seconds: float

    ollama_url: str | None
    ollama_embedding_model: str
    ollama_chat_model: str
    ollama_timeout_seconds: float

    retrieval_top_k: int
    retrieval_min_score: float
    min_rag_hits: int

    sotra_api_key: str | None
    sotra_url: str | None
    sotra_timeout_seconds: float

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
        qdrant_timeout_seconds=float(os.getenv('QDRANT_TIMEOUT_SECONDS', '20')),
        embedding_dimension=int(os.getenv('EMBEDDING_DIMENSION', '1536')),
        vector_distance=cast(
            Literal['cosine', 'dot', 'euclid'],
            os.getenv('VECTOR_DISTANCE', 'cosine'),
        ),
        hybrid_prefetch_limit=int(os.getenv('HYBRID_PREFETCH_LIMIT', '20')),
        embedding_provider=cast(
            Literal['openai', 'ollama'],
            os.getenv('EMBEDDING_PROVIDER', 'openai'),
        ),
        llm_provider=cast(
            Literal['openai', 'ollama'],
            os.getenv('LLM_PROVIDER', 'openai'),
        ),
        openai_api_key=os.getenv('OPENAI_API_KEY') or None,
        openai_base_url=os.getenv('OPENAI_BASE_URL') or None,
        openai_embedding_model=os.getenv('OPENAI_EMBEDDING_MODEL', 'text-embedding-3-small'),
        openai_chat_model=os.getenv('OPENAI_CHAT_MODEL', 'gpt-5-mini'),
        openai_timeout_seconds=float(os.getenv('OPENAI_TIMEOUT_SECONDS', '60')),
        ollama_url=os.getenv('LLM_EMBEDDING_ENDPOINT') or None,
        ollama_embedding_model=os.getenv('LLM_EMBEDDING_MODEL', 'embeddinggemma'),
        ollama_chat_model=os.getenv('LLM_EMBEDDING_MODEL', 'qwen3.5:4b'),
        ollama_timeout_seconds=float(os.getenv('OLLAMA_TIMEOUT_SECONDS', '60')),
        retrieval_top_k=int(os.getenv('RETRIEVAL_TOP_K', '6')),
        retrieval_min_score=float(os.getenv('RETRIEVAL_MIN_SCORE', '0.015')),
        min_rag_hits=int(os.getenv('MIN_RAG_HITS', '1')),
        sotra_api_key=os.getenv('SOTRA_API_KEY') or None,
        sotra_url=os.getenv('SOTRA_URL') or None,
        sotra_timeout_seconds=float(os.getenv('SOTRA_TIMEOUT_SECONDS', '30')),
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
