from __future__ import annotations

from typing import Any

from pydantic import AliasChoices, BaseModel, Field, HttpUrl


class ParsedSection(BaseModel):
    title: str = Field(default='')
    text: str
    language: str | None = None


class ParseUrlRequest(BaseModel):
    url: HttpUrl
    min_chars: int = Field(default=40, ge=0)
    store_in_db: bool = False
    collection: str | None = None
    language: str | None = None


class AskRequest(BaseModel):
    question: str = Field(min_length=1)
    history: list[dict[str, Any]] | None = None
    is_phone_call: bool | None = Field(default=None, validation_alias=AliasChoices('is_phone_call', 'isPhoneCall'))
    collection: str | None = None
    top_k: int | None = Field(default=None, ge=1, le=50, validation_alias=AliasChoices('top_k', 'topK'))
    language: str | None = None


class AskSource(BaseModel):
    source_type: str
    source_url: str = Field(default='')
    title: str = Field(default='')
    source_id: str = Field(default='')
    page: int | None = None
    published_at: str | None = None
    section: str | None = None
    # Retrieval score from Qdrant (1:1 with contexts[i])
    score: float | None = None


class AskResponse(BaseModel):
    answer: str = Field(default='')
    contexts: list[str] = Field(default_factory=list)
    sources: list[AskSource] = Field(default_factory=list)
    # Parallel to contexts/sources when present (legacy-friendly duplicate)
    scores: list[float] = Field(default_factory=list)


class HealthResponse(BaseModel):
    status: str = 'ok'


class CollectionInfo(BaseModel):
    name: str
    points: int | None = None


class CollectionsResponse(BaseModel):
    default: str
    allowed: list[str] = Field(default_factory=list)
    collections: list[CollectionInfo] = Field(default_factory=list)


class DocumentUploadResponse(BaseModel):
    source_id: str
    collection: str
    section_count: int
    chunk_count: int
    filename: str


class DeleteDocumentRequest(BaseModel):
    collection: str
    source_id: str = Field(min_length=1, max_length=200)


class DeleteDocumentResponse(BaseModel):
    deleted: bool
    collection: str
    source_id: str


class DeleteCollectionResponse(BaseModel):
    deleted: bool
    collection: str
