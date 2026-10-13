from __future__ import annotations

from app.clients.qdrant_client import QdrantGateway
from app.core.config import Settings
from app.providers.embedding_factory import create_embedding_provider
from app.services.indexing_service import IndexingService
from app.services.parser_service import ParserService
from app.services.rag_service import RagService
from app.services.reranker import CrossEncoderReranker
from app.services.retrieval_service import RetrievalService
from app.services.scheduler_service import ReparseScheduler
from app.utils.chunking import Chunker
from app.utils.embedding_prompts import resolve_prompt_style
from app.utils.sparse_embedding import SparseEmbeddingProvider, normalize_bm25_language


def _embedding_model_name(settings: Settings) -> str:
    if settings.embedding_provider == 'ollama':
        return settings.ollama_embedding_model
    return settings.openai_embedding_model


class ServiceContainer:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.prompt_style = resolve_prompt_style(
            settings.embedding_prompt_style,
            _embedding_model_name(settings),
        )
        self.bm25_language = normalize_bm25_language(settings.bm25_language)

        self.qdrant = QdrantGateway(settings)
        self.embeddings = create_embedding_provider(settings)
        self.sparse_embeddings = SparseEmbeddingProvider(
            default_language=self.bm25_language,
        )
        self.chunker = Chunker(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )
        self.parser_service = ParserService()
        self.reranker = (
            CrossEncoderReranker(settings.rerank_model)
            if settings.rerank_enabled
            else None
        )
        self.indexing_service = IndexingService(
            qdrant=self.qdrant,
            embeddings=self.embeddings,
            sparse_embeddings=self.sparse_embeddings,
            chunker=self.chunker,
            document_language=settings.document_language,
            prompt_style=self.prompt_style,
            upsert_batch_size=settings.upsert_batch_size,
            bm25_language=self.bm25_language,
        )
        self.retrieval_service = RetrievalService(
            qdrant=self.qdrant,
            embeddings=self.embeddings,
            sparse_embeddings=self.sparse_embeddings,
            hybrid_prefetch_limit=settings.hybrid_prefetch_limit,
            prompt_style=self.prompt_style,
            dense_score_threshold=settings.retrieval_score_threshold,
            candidate_multiplier=settings.retrieval_candidate_multiplier,
            bm25_language=self.bm25_language,
            query_cache_size=settings.query_embedding_cache_size,
            reranker=self.reranker,
            rerank_score_threshold=settings.rerank_score_threshold,
        )
        self.rag_service = RagService(
            retrieval=self.retrieval_service,
            top_k=settings.retrieval_top_k,
            max_context_chunks=settings.max_context_chunks,
            min_rag_hits=settings.min_rag_hits,
            dedup_similarity_threshold=settings.dedup_similarity_threshold,
            merge_adjacent_chunks=settings.merge_adjacent_chunks,
        )
        self.scheduler = ReparseScheduler(
            parser_service=self.parser_service,
            indexing_service=self.indexing_service,
            interval_hours=settings.scheduler_interval_hours,
            urls=settings.reparse_urls,
        )

    async def warm_up(self) -> None:
        """Pay model load cost at boot instead of on the first user question."""
        await self.sparse_embeddings.warm_up([self.bm25_language, 'hsb'])
        if self.reranker is not None:
            await self.reranker.warm_up()

    async def close(self) -> None:
        await self.scheduler.stop()
        import inspect

        close = getattr(self.embeddings, 'close', None)
        if callable(close):
            result = close()
            if inspect.isawaitable(result):
                await result
        await self.qdrant.close()
