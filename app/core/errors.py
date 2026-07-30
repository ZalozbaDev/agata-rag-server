from __future__ import annotations


class ProviderError(Exception):
    def __init__(self, message: str, *, is_timeout: bool = False) -> None:
        super().__init__(message)
        self.is_timeout = is_timeout


class QdrantServiceError(ProviderError):
    pass


class OpenAIServiceError(ProviderError):
    pass


class OllamaServiceError(ProviderError):
    pass


class SotraServiceError(ProviderError):
    pass
