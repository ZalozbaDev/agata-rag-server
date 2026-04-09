from __future__ import annotations

from app.clients.qdrant_client import QdrantGateway
from app.core.config import Settings
from app.providers.openai_provider import OpenAIEmbeddingProvider, OpenAILLMProvider
from app.services.indexing_service import IndexingService
from app.services.parser_service import ParserService
from app.services.rag_service import RagService
from app.services.retrieval_service import RetrievalService
from app.services.scheduler_service import ReparseScheduler
from app.utils.chunking import Chunker


class ServiceContainer:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.qdrant = QdrantGateway(settings)
        self.embeddings = OpenAIEmbeddingProvider(settings)
        self.llm = OpenAILLMProvider(settings)
        self.chunker = Chunker(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )
        self.parser_service = ParserService()
        self.indexing_service = IndexingService(
            qdrant=self.qdrant,
            embeddings=self.embeddings,
            chunker=self.chunker,
        )
        self.retrieval_service = RetrievalService(
            qdrant=self.qdrant,
            embeddings=self.embeddings,
        )
        self.rag_service = RagService(
            retrieval=self.retrieval_service,
            llm=self.llm,
            top_k=settings.retrieval_top_k,
            max_context_chunks=settings.max_context_chunks,
        )
        self.scheduler = ReparseScheduler(
            parser_service=self.parser_service,
            indexing_service=self.indexing_service,
            interval_hours=settings.scheduler_interval_hours,
            urls=settings.reparse_urls,
        )
