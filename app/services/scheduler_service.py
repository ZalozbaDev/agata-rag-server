from __future__ import annotations

import logging

from apscheduler.schedulers.asyncio import AsyncIOScheduler

from app.services.indexing_service import IndexingService
from app.services.parser_service import ParserService

logger = logging.getLogger(__name__)


class ReparseScheduler:
    def __init__(
        self,
        parser_service: ParserService,
        indexing_service: IndexingService,
        interval_hours: int,
        urls: list[str],
    ) -> None:
        self._parser_service = parser_service
        self._indexing_service = indexing_service
        self._interval_hours = interval_hours
        self._urls = urls
        self._scheduler = AsyncIOScheduler()

    def start(self) -> None:
        if not self._urls:
            logger.info('Scheduler gestartet, aber keine REPARSE_URLS konfiguriert.')
            return

        self._scheduler.add_job(
            self._run_job,
            trigger='interval',
            hours=self._interval_hours,
            id='reparse_urls',
            replace_existing=True,
            max_instances=1,
            coalesce=True,
        )
        self._scheduler.start()
        logger.info('Scheduler gestartet. Intervall: %s Stunden', self._interval_hours)

    async def stop(self) -> None:
        if self._scheduler.running:
            self._scheduler.shutdown(wait=False)

    async def _run_job(self) -> None:
        logger.info('Starte geplantes Re-Parsing von %s URLs', len(self._urls))
        for url in self._urls:
            try:
                sections = await self._parser_service.parse_url(url, min_chars=40)
            except Exception:
                logger.exception('Geplantes Re-Parsing fehlgeschlagen für URL: %s', url)
                continue

            source_id = f'url:{url}'
            await self._indexing_service.store_sections(
                source_id=source_id,
                source_type='url',
                sections=sections,
            )

        logger.info('Geplantes Re-Parsing abgeschlossen.')
