"""End-to-end integration test: 5-turn coding assistant conversation."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from hcma.agents.task_agent import TaskAgent
from hcma.memory.episodic_buffer import EpisodicBuffer


_TURNS = [
    "How do I reverse a list in Python?",        # 2 entries (no debug kw)
    "I have a bug in my code, it throws an IndexError",  # 3 entries (bug, error)
    "What is a decorator?",                      # 2 entries
    "Fix this error: TypeError on line 42",      # 3 entries (fix, error)
    "Explain list comprehensions",               # 2 entries
]
# Total minimum: 2+3+2+3+2 = 12; debug entries from turns 2 and 4


def _mock_response(content: str = "Mock assistant reply.") -> MagicMock:
    msg = MagicMock()
    msg.content = content
    resp = MagicMock()
    resp.message = msg
    return resp


def _run_session() -> tuple[EpisodicBuffer, TaskAgent]:
    buf = EpisodicBuffer(":memory:", capacity=50)
    agent = TaskAgent(buf, session_id="integration_session_001")
    with patch.object(agent._client, "chat", return_value=_mock_response()):
        for turn in _TURNS:
            agent.run(turn)
    return buf, agent


class TestIntegration:
    def setup_method(self):
        self.buf, self.agent = _run_session()
        self.all_entries = self.buf.read_all_raw()

    # --- entry count ---

    def test_buffer_has_at_least_12_entries(self):
        assert self.buf.get_count() >= 12, (
            f"Expected >=12 entries, got {self.buf.get_count()}"
        )

    def test_buffer_not_at_capacity(self):
        assert not self.buf.is_at_capacity(), (
            f"Buffer unexpectedly at capacity: {self.buf.get_count()}/{self.buf.capacity}"
        )

    # --- session_id ---

    def test_all_entries_have_correct_session_id(self):
        for entry in self.all_entries:
            assert entry.session_id == "integration_session_001", (
                f"Entry {entry.id[:8]} has wrong session_id: {entry.session_id!r}"
            )

    # --- status ---

    def test_all_entries_are_raw(self):
        for entry in self.all_entries:
            assert entry.status == "raw", (
                f"Entry {entry.id[:8]} has status {entry.status!r}, expected 'raw'"
            )

    # --- debug entries from error-keyword turns ---

    def test_debug_entries_exist(self):
        debug_entries = [e for e in self.all_entries if "debug" in e.tags]
        assert len(debug_entries) >= 2, (
            f"Expected >=2 debug entries, got {len(debug_entries)}"
        )

    def test_debug_entries_have_error_pattern_tag(self):
        debug_entries = [e for e in self.all_entries if "debug" in e.tags]
        for entry in debug_entries:
            assert "error_pattern" in entry.tags

    def test_debug_entries_have_high_importance(self):
        debug_entries = [e for e in self.all_entries if "debug" in e.tags]
        for entry in debug_entries:
            assert entry.importance == 0.8

    # --- tag coverage ---

    def test_user_query_entries_exist(self):
        uq = [e for e in self.all_entries if "user_query" in e.tags]
        assert len(uq) == 5  # one per turn

    def test_assistant_response_entries_exist(self):
        ar = [e for e in self.all_entries if "assistant_response" in e.tags]
        assert len(ar) == 5  # one per turn

    # --- conversation history ---

    def test_conversation_history_length(self):
        # 5 turns × 2 messages (user + assistant) = 10
        assert len(self.agent.get_conversation_history()) == 10

    # --- summary table (printed, not asserted) ---

    def test_print_summary_table(self, capsys):
        header = f"{'ID':10} {'TAGS':30} {'IMP':5} CONTENT"
        print("\n" + header)
        print("-" * 80)
        for entry in sorted(self.all_entries, key=lambda e: e.timestamp):
            tags_str = ", ".join(entry.tags)
            print(
                f"{entry.id[:8]:<10} {tags_str:<30} {entry.importance:<5} "
                f"{entry.content[:60]}"
            )
        captured = capsys.readouterr()
        assert "user_query" in captured.out
        assert "assistant_response" in captured.out
        assert "debug" in captured.out
