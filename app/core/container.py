from __future__ import annotations

from app.clients.qdrant_client import QdrantGateway
from app.core.config import Settings
from app.providers.embedding_factory import create_embedding_provider
from app.providers.sotra_provider import SotraProvider
from app.services.indexing_service import IndexingService
from app.services.parser_service import ParserService
from app.services.rag_service import RagService
from app.services.retrieval_service import RetrievalService
from app.services.scheduler_service import ReparseScheduler
from app.utils.chunking import Chunker
from app.utils.sparse_embedding import SparseEmbeddingProvider


class ServiceContainer:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.qdrant = QdrantGateway(settings)
        self.embeddings = create_embedding_provider(settings)
        self.sparse_embeddings = SparseEmbeddingProvider()
        self.sotra = SotraProvider(settings)
        self.chunker = Chunker(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )
        self.parser_service = ParserService()
        self.indexing_service = IndexingService(
            qdrant=self.qdrant,
            embeddings=self.embeddings,
            sparse_embeddings=self.sparse_embeddings,
            chunker=self.chunker,
        )
        self.retrieval_service = RetrievalService(
            qdrant=self.qdrant,
            embeddings=self.embeddings,
            sparse_embeddings=self.sparse_embeddings,
            sotra=self.sotra,
            hybrid_prefetch_limit=settings.hybrid_prefetch_limit,
        )
        self.rag_service = RagService(
            retrieval=self.retrieval_service,
            top_k=settings.retrieval_top_k,
            max_context_chunks=settings.max_context_chunks,
            retrieval_min_score=settings.retrieval_min_score,
            min_rag_hits=settings.min_rag_hits,
        )
        self.scheduler = ReparseScheduler(
            parser_service=self.parser_service,
            indexing_service=self.indexing_service,
            interval_hours=settings.scheduler_interval_hours,
            urls=settings.reparse_urls,
        )

    async def close(self) -> None:
        await self.scheduler.stop()
        await self.sotra.close()
        import inspect

        close = getattr(self.embeddings, 'close', None)
        if callable(close):
            result = close()
            if inspect.isawaitable(result):
                await result
        await self.qdrant.close()
