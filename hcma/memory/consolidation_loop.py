"""Background thread that triggers consolidation when the episodic buffer fills up."""

from __future__ import annotations

import logging
import threading

from hcma.agents.consolidation_agent import ConsolidationAgent
from hcma.config import settings
from hcma.memory.episodic_buffer import EpisodicBuffer
from hcma.memory.ltm_store import LTMStore

logger = logging.getLogger(__name__)


class ConsolidationLoop:
    def __init__(
        self,
        buffer: EpisodicBuffer,
        ltm: LTMStore,
        check_interval_seconds: int = 30,
    ) -> None:
        self.buffer = buffer
        self.ltm = ltm
        self.check_interval_seconds = check_interval_seconds
        self._agent = ConsolidationAgent(buffer, ltm)
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def start(self) -> None:
        self._stop_event.clear()
        self._thread.start()
        logger.info("Consolidation loop started (interval: %ds)", self.check_interval_seconds)

    def stop(self) -> None:
        self._stop_event.set()
        self._thread.join(timeout=10)
        logger.info("Consolidation loop stopped")

    def _loop(self) -> None:
        while not self._stop_event.is_set():
            self._stop_event.wait(timeout=self.check_interval_seconds)
            if self._stop_event.is_set():
                break
            try:
                trigger_count = int(
                    self.buffer.capacity * settings.CONSOLIDATION_TRIGGER_RATIO
                )
                current = self.buffer.get_count()
                if current >= trigger_count:
                    logger.info(
                        "Buffer at %d/%d (trigger=%d) — running consolidation",
                        current, self.buffer.capacity, trigger_count,
                    )
                    result = self._agent.run()
                    logger.info(
                        "Consolidation complete: %d promoted, %d compressed, "
                        "%d discarded, %d contradictions, took %.2fs",
                        result.promoted, result.compressed, result.discarded,
                        result.contradictions_found, result.duration_seconds,
                    )
                else:
                    logger.debug(
                        "Buffer at %d/%d, no consolidation needed",
                        current, self.buffer.capacity,
                    )
            except Exception:
                logger.exception("ConsolidationLoop._loop: unexpected error, continuing")
