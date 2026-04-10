from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import AliasChoices, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file='.env',
        env_file_encoding='utf-8',
        case_sensitive=False,
        extra='ignore',
    )

    app_name: str = 'rag-server'
    environment: Literal['local', 'dev', 'prod'] = 'local'
    log_level: str = 'INFO'
    api_prefix: str = ''

    qdrant_url: str = 'http://qdrant:6333'
    qdrant_api_key: str | None = None
    qdrant_collection: str = 'documents'
    embedding_dimension: int = 1536
    vector_distance: Literal['cosine', 'dot', 'euclid'] = 'cosine'

    llm_provider: Literal['openai'] = 'openai'
    embedding_provider: Literal['openai'] = 'openai'
    openai_api_key: str | None = None
    openai_base_url: str | None = None
    openai_chat_model: str = 'gpt-5-mini'
    openai_embedding_model: str = 'text-embedding-3-small'

    retrieval_top_k: int = 6
    retrieval_min_score: float = 0.55
    min_rag_hits: int = 1

    sotra_api_key: str | None = None
    sotra_url: str | None = None
    sotra_timeout_seconds: float = 30.0    

    chunk_size: int = 1200
    chunk_overlap: int = 150
    max_context_chunks: int = 8

    scheduler_enabled: bool = True
    scheduler_interval_hours: int = 24
    reparse_urls: list[str] = Field(default_factory=list)

    cors_origins: list[str] = Field(default_factory=lambda: ['*'])
    cors_allow_credentials: bool = False

    @field_validator('cors_origins', mode='before')
    @classmethod
    def _parse_cors_origins(cls, value: object) -> list[str]:
        if value is None:
            return ['*']
        if isinstance(value, list):
            return [str(v).strip() for v in value if str(v).strip()]
        if isinstance(value, str):
            cleaned = value.strip()
            if not cleaned:
                return ['*']
            if cleaned.startswith('['):
                import json

                loaded = json.loads(cleaned)
                return [str(v).strip() for v in loaded if str(v).strip()]
            return [part.strip() for part in cleaned.split(',') if part.strip()]
        raise TypeError('CORS_ORIGINS must be a JSON array or comma-separated string')


@lru_cache
def get_settings() -> Settings:
    return Settings()