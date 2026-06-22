"""Unit tests for TaskAgent — all LLM calls are mocked."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from hcma.agents.task_agent import TaskAgent
from hcma.memory.episodic_buffer import EpisodicBuffer
from hcma.memory.ltm_store import LTMStore
from hcma.schemas.memory_types import LTMMemory


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SESSION = "test_session_001"
_LLM_REPLY = "Here is the answer to your question."


def _buf() -> EpisodicBuffer:
    return EpisodicBuffer(":memory:", capacity=50)


def _agent(buf: EpisodicBuffer | None = None) -> TaskAgent:
    return TaskAgent(buf or _buf(), session_id=_SESSION)


def _mock_chat_response(content: str = _LLM_REPLY) -> MagicMock:
    """Build a fake ollama ChatResponse object."""
    msg = MagicMock()
    msg.content = content
    resp = MagicMock()
    resp.message = msg
    return resp


# ---------------------------------------------------------------------------
# run() — basic behaviour
# ---------------------------------------------------------------------------

class TestRunBasic:
    def test_returns_llm_response_string(self):
        buf = _buf()
        agent = _agent(buf)
        with patch.object(agent._client, "chat", return_value=_mock_chat_response()) as mock_chat:
            result = agent.run("How do I reverse a list in Python?")
        assert result == _LLM_REPLY

    def test_chat_called_with_correct_model(self):
        from hcma.config import settings
        agent = _agent()
        with patch.object(agent._client, "chat", return_value=_mock_chat_response()) as mock_chat:
            agent.run("test input")
        call_kwargs = mock_chat.call_args
        assert call_kwargs.kwargs.get("model") == settings.OLLAMA_MODEL or \
               call_kwargs.args[0] == settings.OLLAMA_MODEL

    def test_system_prompt_injected_not_stored(self):
        agent = _agent()
        with patch.object(agent._client, "chat", return_value=_mock_chat_response()) as mock_chat:
            agent.run("hello")
        messages_sent = mock_chat.call_args.kwargs.get("messages") or mock_chat.call_args.args[1]
        # System message must appear in what was sent to Ollama
        assert messages_sent[0]["role"] == "system"
        assert "coding assistant" in messages_sent[0]["content"]
        # But NOT stored in conversation_history
        for msg in agent.conversation_history:
            assert msg["role"] != "system"


# ---------------------------------------------------------------------------
# run() — episodic entry creation
# ---------------------------------------------------------------------------

class TestEpisodicEntries:
    def test_normal_input_creates_two_entries(self):
        buf = _buf()
        agent = _agent(buf)
        with patch.object(agent._client, "chat", return_value=_mock_chat_response()):
            agent.run("How do I use a list comprehension?")
        assert buf.get_count() == 2

    def test_error_keyword_creates_three_entries(self):
        buf = _buf()
        agent = _agent(buf)
        with patch.object(agent._client, "chat", return_value=_mock_chat_response()):
            agent.run("I have an error in my code")
        assert buf.get_count() == 3

    @pytest.mark.parametrize("keyword", ["error", "bug", "fix", "crash", "exception", "traceback", "fail"])
    def test_each_debug_keyword_triggers_third_entry(self, keyword):
        buf = _buf()
        agent = _agent(buf)
        with patch.object(agent._client, "chat", return_value=_mock_chat_response()):
            agent.run(f"My code has a {keyword} I cannot solve")
        assert buf.get_count() == 3

    def test_debug_keyword_uppercase_triggers_third_entry(self):
        buf = _buf()
        agent = _agent(buf)
        with patch.object(agent._client, "chat", return_value=_mock_chat_response()):
            agent.run("There is an ERROR in my function")
        assert buf.get_count() == 3

    def test_user_query_entry_content(self):
        buf = _buf()
        agent = _agent(buf)
        user_input = "How do I read a file?"
        with patch.object(agent._client, "chat", return_value=_mock_chat_response()):
            agent.run(user_input)
        raw = buf.read_all_raw()
        user_entries = [e for e in raw if "user_query" in e.tags]
        assert len(user_entries) == 1
        assert f"User asked about: {user_input}" in user_entries[0].content

    def test_assistant_response_entry_content(self):
        buf = _buf()
        agent = _agent(buf)
        with patch.object(agent._client, "chat", return_value=_mock_chat_response(_LLM_REPLY)):
            agent.run("Explain decorators")
        raw = buf.read_all_raw()
        resp_entries = [e for e in raw if "assistant_response" in e.tags]
        assert len(resp_entries) == 1
        assert _LLM_REPLY[:300] in resp_entries[0].content

    def test_debug_entry_tags(self):
        buf = _buf()
        agent = _agent(buf)
        with patch.object(agent._client, "chat", return_value=_mock_chat_response()):
            agent.run("I have a bug in my code")
        raw = buf.read_all_raw()
        debug_entries = [e for e in raw if "debug" in e.tags]
        assert len(debug_entries) == 1
        assert "error_pattern" in debug_entries[0].tags
        assert debug_entries[0].importance == 0.8

    def test_entries_use_correct_session_id(self):
        buf = _buf()
        agent = _agent(buf)
        with patch.object(agent._client, "chat", return_value=_mock_chat_response()):
            agent.run("test question")
        for entry in buf.read_all_raw():
            assert entry.session_id == _SESSION


# ---------------------------------------------------------------------------
# conversation_history
# ---------------------------------------------------------------------------

class TestConversationHistory:
    def test_history_grows_after_each_run(self):
        agent = _agent()
        with patch.object(agent._client, "chat", return_value=_mock_chat_response()):
            agent.run("first question")
            agent.run("second question")
        # Two turns = 4 messages (user + assistant each time)
        assert len(agent.conversation_history) == 4

    def test_history_alternates_roles(self):
        agent = _agent()
        with patch.object(agent._client, "chat", return_value=_mock_chat_response()):
            agent.run("question one")
        assert agent.conversation_history[0]["role"] == "user"
        assert agent.conversation_history[1]["role"] == "assistant"

    def test_history_contains_user_input(self):
        agent = _agent()
        user_input = "What is a generator?"
        with patch.object(agent._client, "chat", return_value=_mock_chat_response()):
            agent.run(user_input)
        assert agent.conversation_history[0]["content"] == user_input

    def test_history_contains_assistant_reply(self):
        agent = _agent()
        with patch.object(agent._client, "chat", return_value=_mock_chat_response(_LLM_REPLY)):
            agent.run("explain generators")
        assert agent.conversation_history[1]["content"] == _LLM_REPLY

    def test_get_conversation_history_returns_copy(self):
        agent = _agent()
        with patch.object(agent._client, "chat", return_value=_mock_chat_response()):
            agent.run("test")
        history = agent.get_conversation_history()
        history.clear()
        assert len(agent.conversation_history) == 2

    def test_prior_history_sent_to_llm(self):
        agent = _agent()
        with patch.object(agent._client, "chat", return_value=_mock_chat_response()) as mock_chat:
            agent.run("first")
            agent.run("second")
        last_call_messages = mock_chat.call_args.kwargs.get("messages") or mock_chat.call_args.args[1]
        # system + user("first") + assistant(reply) + user("second") = 4
        assert len(last_call_messages) == 4


# ---------------------------------------------------------------------------
# clear_history
# ---------------------------------------------------------------------------

class TestClearHistory:
    def test_clear_history_empties_list(self):
        agent = _agent()
        with patch.object(agent._client, "chat", return_value=_mock_chat_response()):
            agent.run("a question")
        agent.clear_history()
        assert agent.conversation_history == []

    def test_clear_history_does_not_affect_buffer(self):
        buf = _buf()
        agent = _agent(buf)
        with patch.object(agent._client, "chat", return_value=_mock_chat_response()):
            agent.run("a question")
        count_before = buf.get_count()
        agent.clear_history()
        assert buf.get_count() == count_before

    def test_run_after_clear_starts_fresh(self):
        agent = _agent()
        with patch.object(agent._client, "chat", return_value=_mock_chat_response()) as mock_chat:
            agent.run("first question")
            agent.clear_history()
            agent.run("second question")
        last_messages = mock_chat.call_args.kwargs.get("messages") or mock_chat.call_args.args[1]
        # Only system + one user message — no leftover history
        assert len(last_messages) == 2


# ---------------------------------------------------------------------------
# LLM failure path
# ---------------------------------------------------------------------------

class TestLLMFailure:
    def test_llm_error_returns_fallback_string(self):
        agent = _agent()
        with patch.object(agent._client, "chat", side_effect=RuntimeError("connection refused")):
            result = agent.run("any question")
        assert result == "I encountered an error. Please try again."

    def test_llm_error_writes_zero_entries(self):
        buf = _buf()
        agent = _agent(buf)
        with patch.object(agent._client, "chat", side_effect=RuntimeError("timeout")):
            agent.run("any question")
        assert buf.get_count() == 0


# ---------------------------------------------------------------------------
# LTM retrieval injection
# ---------------------------------------------------------------------------

def _fake_ltm(memories: list[LTMMemory]) -> MagicMock:
    ltm = MagicMock(spec=LTMStore)
    ltm.search_semantic.return_value = memories
    return ltm


def _fake_memory(content: str) -> LTMMemory:
    import time as _time
    return LTMMemory(
        content=content,
        source_episode_ids=["ep-1"],
        created_at=_time.time(),
        last_accessed=_time.time(),
    )


class TestLTMRetrieval:
    def test_memory_context_injected_into_system_prompt(self):
        ltm = _fake_ltm([_fake_memory("User prefers type hints in all functions")])
        agent = TaskAgent(_buf(), session_id=_SESSION, ltm=ltm)
        with patch.object(agent._client, "chat", return_value=_mock_chat_response()) as mock_chat:
            agent.run("How do I write a function?")
        messages = mock_chat.call_args.kwargs.get("messages") or mock_chat.call_args.args[1]
        system_content = messages[0]["content"]
        assert "coding assistant" in system_content
        assert "User prefers type hints" in system_content

    def test_memory_context_not_stored_in_conversation_history(self):
        ltm = _fake_ltm([_fake_memory("User prefers type hints in all functions")])
        agent = TaskAgent(_buf(), session_id=_SESSION, ltm=ltm)
        with patch.object(agent._client, "chat", return_value=_mock_chat_response()):
            agent.run("How do I write a function?")
        for msg in agent.conversation_history:
            assert "Relevant memories" not in msg["content"]

    def test_no_ltm_system_prompt_unchanged(self):
        agent = TaskAgent(_buf(), session_id=_SESSION, ltm=None)
        with patch.object(agent._client, "chat", return_value=_mock_chat_response()) as mock_chat:
            agent.run("How do I write a function?")
        messages = mock_chat.call_args.kwargs.get("messages") or mock_chat.call_args.args[1]
        from hcma.agents.task_agent import _SYSTEM_PROMPT
        assert messages[0]["content"] == _SYSTEM_PROMPT

    def test_empty_ltm_results_leave_system_prompt_unchanged(self):
        ltm = _fake_ltm([])
        agent = TaskAgent(_buf(), session_id=_SESSION, ltm=ltm)
        with patch.object(agent._client, "chat", return_value=_mock_chat_response()) as mock_chat:
            agent.run("How do I write a function?")
        messages = mock_chat.call_args.kwargs.get("messages") or mock_chat.call_args.args[1]
        from hcma.agents.task_agent import _SYSTEM_PROMPT
        assert messages[0]["content"] == _SYSTEM_PROMPT

    def test_ltm_search_failure_falls_back_gracefully(self):
        ltm = MagicMock(spec=LTMStore)
        ltm.search_semantic.side_effect = RuntimeError("qdrant unavailable")
        agent = TaskAgent(_buf(), session_id=_SESSION, ltm=ltm)
        with patch.object(agent._client, "chat", return_value=_mock_chat_response()) as mock_chat:
            result = agent.run("test question")
        assert result == _LLM_REPLY
        messages = mock_chat.call_args.kwargs.get("messages") or mock_chat.call_args.args[1]
        from hcma.agents.task_agent import _SYSTEM_PROMPT
        assert messages[0]["content"] == _SYSTEM_PROMPT

    def test_multiple_memories_all_injected(self):
        ltm = _fake_ltm([
            _fake_memory("User prefers pytest over unittest"),
            _fake_memory("Project uses Python 3.11"),
        ])
        agent = TaskAgent(_buf(), session_id=_SESSION, ltm=ltm)
        with patch.object(agent._client, "chat", return_value=_mock_chat_response()) as mock_chat:
            agent.run("How do I run tests?")
        messages = mock_chat.call_args.kwargs.get("messages") or mock_chat.call_args.args[1]
        system_content = messages[0]["content"]
        assert "User prefers pytest" in system_content
        assert "Python 3.11" in system_content

    def test_ltm_search_called_with_user_query(self):
        ltm = _fake_ltm([])
        agent = TaskAgent(_buf(), session_id=_SESSION, ltm=ltm)
        user_query = "How do I use async/await?"
        with patch.object(agent._client, "chat", return_value=_mock_chat_response()):
            agent.run(user_query)
        ltm.search_semantic.assert_called_once_with(user_query, top_k=3)
