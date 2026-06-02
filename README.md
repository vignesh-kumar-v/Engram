# Hierarchical Consolidation Memory Agent (HCMA)

A memory system for coding assistants that consolidates short-term episodic events into long-term structured knowledge.

## Architecture

```
Episodic Buffer (SQLite, in-session)
        │
        ▼  consolidation (importance threshold)
LTM Store (SQLite index + Qdrant vectors)
        │
        ▼
TaskAgent (LangGraph) queries LTM to inform coding assistance
```

## Project Structure

```
hcma/
├── config/settings.py       — tuneable constants
├── schemas/memory_types.py  — EpisodicEntry and LTMMemory dataclasses
├── memory/episodic_buffer.py — short-term buffer (Week 2)
├── memory/ltm_store.py      — long-term store (Week 2)
├── agents/task_agent.py     — LangGraph agent (Week 3)
├── tests/                   — pytest test suite
└── scripts/run_agent.py     — CLI entry point
```

## Quickstart

```bash
pip install -r requirements.txt
pytest hcma/tests/
```

## Configuration

All tuneable constants live in `hcma/config/settings.py`. Override via environment variables using `python-dotenv` (`.env` file in project root).

## Roadmap

- **Week 1** (current): project scaffold, schemas, settings
- **Week 2**: EpisodicBuffer (SQLite) + LTMStore (Qdrant)
- **Week 3**: TaskAgent (LangGraph) + consolidation logic
