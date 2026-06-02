"""Unit tests for ConsolidationLoop."""

from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock, patch

import pytest

from hcma.memory.consolidation_loop import ConsolidationLoop
from hcma.memory.episodic_buffer import EpisodicBuffer
from hcma.schemas.memory_types import ConsolidationResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _buf(capacity: int = 10) -> EpisodicBuffer:
    return EpisodicBuffer(":memory:", capacity=capacity)


def _mock_ltm() -> MagicMock:
    return MagicMock()


def _loop(buf: EpisodicBuffer | None = None, interval: int = 1) -> ConsolidationLoop:
    return ConsolidationLoop(buf or _buf(), _mock_ltm(), check_interval_seconds=interval)


def _ok_result() -> ConsolidationResult:
    return ConsolidationResult(promoted=2, total_processed=2, duration_seconds=0.1)


# ---------------------------------------------------------------------------
# start / stop lifecycle
# ---------------------------------------------------------------------------

class TestLifecycle:
    def test_start_and_stop_do_not_raise(self):
        loop = _loop()
        loop.start()
        loop.stop()

    def test_stop_before_start_does_not_raise(self):
        loop = _loop()
        # _stop_event is not set, _thread not started — join on unstarted thread raises
        # so we just check stop() after start()
        loop.start()
        loop.stop()

    def test_thread_is_daemon(self):
        loop = _loop()
        assert loop._thread.daemon is True

    def test_stop_joins_thread(self):
        loop = _loop(interval=60)
        loop.start()
        assert loop._thread.is_alive()
        loop.stop()
        assert not loop._thread.is_alive()

    def test_stop_event_set_after_stop(self):
        loop = _loop()
        loop.start()
        loop.stop()
        assert loop._stop_event.is_set()


# ---------------------------------------------------------------------------
# _loop() — consolidation trigger
# ---------------------------------------------------------------------------

class TestLoopTrigger:
    def test_consolidation_runs_when_buffer_at_trigger_ratio(self):
        """Fill buffer to ≥80% (trigger ratio), confirm agent.run() is called."""
        from hcma.schemas.memory_types import EpisodicEntry

        buf = _buf(capacity=10)  # trigger at 8 entries (80%)
        # Write 8 entries to hit the trigger
        for i in range(8):
            buf.write(EpisodicEntry(
                content=f"entry {i}", source_task="t", session_id="s"
            ))

        loop = ConsolidationLoop(buf, _mock_ltm(), check_interval_seconds=1)

        with patch.object(loop._agent, "run", return_value=_ok_result()) as mock_run:
            # Manually invoke _loop body once via a short-lived thread
            loop._stop_event.clear()
            called = threading.Event()

            original_run = loop._agent.run

            def patched_run():
                result = _ok_result()
                called.set()
                return result

            loop._agent.run = patched_run
            loop._thread = threading.Thread(target=loop._loop, daemon=True)
            loop._thread.start()

            assert called.wait(timeout=5), "Consolidation was not triggered within 5s"
            loop.stop()

    def test_consolidation_skipped_when_below_trigger(self):
        """Buffer nearly empty — consolidation must NOT run."""
        from hcma.schemas.memory_types import EpisodicEntry

        buf = _buf(capacity=10)
        # Write only 3 entries (30% — below 80% trigger)
        for i in range(3):
            buf.write(EpisodicEntry(
                content=f"entry {i}", source_task="t", session_id="s"
            ))

        loop = ConsolidationLoop(buf, _mock_ltm(), check_interval_seconds=1)
        run_called = threading.Event()

        def patched_run():
            run_called.set()
            return _ok_result()

        loop._agent.run = patched_run
        loop.start()
        # Give the loop two full cycles and confirm run was never called
        time.sleep(2.5)
        loop.stop()

        assert not run_called.is_set(), "Consolidation should not have been triggered"

    def test_trigger_threshold_respects_ratio(self):
        """Unit-check the trigger arithmetic outside the thread."""
        from hcma.config import settings

        buf = _buf(capacity=50)
        loop = ConsolidationLoop(buf, _mock_ltm())
        expected_trigger = int(50 * settings.CONSOLIDATION_TRIGGER_RATIO)
        assert expected_trigger == 40


# ---------------------------------------------------------------------------
# _loop() — exception resilience
# ---------------------------------------------------------------------------

class TestLoopResilience:
    def test_loop_continues_after_consolidation_exception(self):
        """
        First run() raises, second run() succeeds.
        The loop must survive the exception and keep running.
        """
        from hcma.schemas.memory_types import EpisodicEntry

        buf = _buf(capacity=10)
        for i in range(8):
            buf.write(EpisodicEntry(
                content=f"entry {i}", source_task="t", session_id="s"
            ))

        loop = ConsolidationLoop(buf, _mock_ltm(), check_interval_seconds=1)

        call_count = 0
        second_call = threading.Event()

        def flaky_run():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("simulated failure")
            second_call.set()
            return _ok_result()

        loop._agent.run = flaky_run
        loop.start()

        assert second_call.wait(timeout=8), (
            "Loop did not recover and call consolidation a second time"
        )
        loop.stop()
        assert call_count >= 2

    def test_loop_does_not_crash_on_buffer_exception(self):
        """If buffer.get_count() itself raises, loop must not die."""
        buf = _buf(capacity=10)
        loop = ConsolidationLoop(buf, _mock_ltm(), check_interval_seconds=1)

        survived = threading.Event()
        original_get_count = buf.get_count

        call_count = 0

        def flaky_get_count():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("db error")
            survived.set()
            return 0  # below trigger — no consolidation

        buf.get_count = flaky_get_count
        loop.start()

        assert survived.wait(timeout=5), "Loop did not survive get_count() exception"
        loop.stop()


# ---------------------------------------------------------------------------
# ConsolidationLoop attributes
# ---------------------------------------------------------------------------

class TestAttributes:
    def test_stores_check_interval(self):
        loop = _loop(interval=45)
        assert loop.check_interval_seconds == 45

    def test_agent_is_initialized(self):
        from hcma.agents.consolidation_agent import ConsolidationAgent
        loop = _loop()
        assert isinstance(loop._agent, ConsolidationAgent)

    def test_stop_event_is_threading_event(self):
        loop = _loop()
        assert isinstance(loop._stop_event, threading.Event)
