from __future__ import annotations

from typing import Any

from pydantic import AliasChoices, BaseModel, Field, HttpUrl


class ParsedSection(BaseModel):
    title: str = Field(default='')
    text: str


class ParseUrlRequest(BaseModel):
    url: HttpUrl
    min_chars: int = Field(default=40, ge=0)
    store_in_db: bool = False


class AskRequest(BaseModel):
    question: str = Field(min_length=1)
    history: list[dict[str, Any]] | None = None
    is_phone_call: bool | None = Field(default=None, validation_alias=AliasChoices('is_phone_call', 'isPhoneCall'))


class AskSource(BaseModel):
    source_type: str
    source_url: str = Field(default='')
    title: str = Field(default='')


class AskResponse(BaseModel):
    answer: str = Field(default='')
    contexts: list[str] = Field(default_factory=list)
    sources: list[AskSource] = Field(default_factory=list)


class HealthResponse(BaseModel):
    status: str = 'ok'
