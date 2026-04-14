from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, HttpUrl


class ParsedSection(BaseModel):
    title: str = Field(default='')
    text: str


class ParseUrlRequest(BaseModel):
    url: HttpUrl
    min_chars: int = Field(default=40, ge=0)
    store_in_db: bool = False


class AskRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    question: str = Field(min_length=1)
    history: list[str] = Field(default_factory=list)
    is_phone_call: bool = Field(default=False, alias='isPhoneCall')


class AskSource(BaseModel):
    source_type: str
    source_url: str = Field(default='')
    title: str = Field(default='')


class AskResponse(BaseModel):
    answer: str
    sources: list[AskSource] = Field(default_factory=list)
    source_strategy: str = Field(default='rag')


class ParseResponse(BaseModel):
    items: list[ParsedSection]


class HealthResponse(BaseModel):
    status: str = 'ok'