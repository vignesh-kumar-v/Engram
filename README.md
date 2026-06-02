# Engram

**A coding assistant with biologically-inspired hierarchical memory — episodic buffering, LLM-driven consolidation, and long-term semantic retrieval.**

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)
![LangGraph](https://img.shields.io/badge/LangGraph-0.2+-orange)
![Qdrant](https://img.shields.io/badge/Qdrant-1.9+-red)

---

## Motivation

Most LLM-based coding assistants are stateless — every session starts from scratch, discarding everything learned about a user's codebase, preferences, and recurring errors. Existing memory solutions like MemGPT treat memory as a retrieval and storage problem: compress everything into a vector store and search it. Engram treats memory as a **reasoning problem**: raw observations are buffered episodically, then a dedicated LLM agent decides what to promote, compress, or discard before anything reaches long-term storage. This design is inspired by how biological memory consolidates during sleep — not storing everything indiscriminately, but extracting signal from noise.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         User Session                            │
└──────────────────────────────┬──────────────────────────────────┘
                               │ user query
                               ▼
                    ┌──────────────────┐
                    │   Task Agent     │  ← LLM (qwen2.5:7b)
                    │  (task_agent.py) │    conversation history
                    └────────┬─────────┘
                             │ writes EpisodicEntry
                             ▼
                    ┌──────────────────┐
                    │ Episodic Buffer  │  ← SQLite, capped at 50
                    │(episodic_buffer) │    entries (raw / promoted
                    └────────┬─────────┘    / discarded)
                             │
              ───────────────┴──────────────────
              │  trigger: buffer ≥ 80% capacity │
              ──────────────────────────────────
                             │
                             ▼
                    ┌──────────────────┐
                    │Consolidation     │  ← LLM decides per entry:
                    │    Agent         │    promote / compress /
                    │(consolidation_   │    discard
                    │  agent.py)       │    + contradiction detection
                    └────────┬─────────┘
                             │ writes LTMMemory
                             ▼
                    ┌──────────────────┐
                    │   LTM Store      │  ← SQLite (metadata)
                    │  (ltm_store.py)  │    Qdrant (768-dim vectors)
                    └──────────────────┘    nomic-embed-text
```

The **Consolidation Loop** runs as a background daemon thread and fires automatically when the episodic buffer reaches the trigger threshold.

---

## Core Components

| Component | File | Description |
|---|---|---|
| `EpisodicEntry` | `hcma/schemas/memory_types.py` | Dataclass for a single raw observation with tags, importance, and status lifecycle |
| `LTMMemory` | `hcma/schemas/memory_types.py` | Consolidated long-term memory with confidence, access count, and vector ID |
| `EpisodicBuffer` | `hcma/memory/episodic_buffer.py` | SQLite-backed bounded buffer; stores raw entries before consolidation |
| `LTMStore` | `hcma/memory/ltm_store.py` | Dual-backend store: SQLite for structured metadata, Qdrant for semantic vector search |
| `TaskAgent` | `hcma/agents/task_agent.py` | Ollama-backed chat agent that records episodic observations per turn |
| `ConsolidationAgent` | `hcma/agents/consolidation_agent.py` | LLM-driven agent that promotes, compresses, or discards episodic entries into LTM |
| `ConsolidationLoop` | `hcma/memory/consolidation_loop.py` | Background thread that monitors buffer fill and triggers consolidation automatically |
| `BenchmarkRunner` | `benchmark/runner.py` | Runs all LHMBench scenarios across all systems |
| `BenchmarkEvaluator` | `benchmark/evaluator.py` | Aggregates results, prints the report table, saves JSON |

---

## How It Works

- **Working memory** — The `TaskAgent` maintains a conversation history for the current session and generates structured `EpisodicEntry` records tagging user queries, assistant responses, and debugging interactions.

- **Episodic buffer** — Entries accumulate in a SQLite-backed buffer capped at 50 entries. Each entry carries an importance score (0–1), status (`raw` → `promoted` / `discarded`), and free-form tags. The buffer is queryable by tag, status, and timestamp.

- **LLM-driven consolidation** — When the buffer reaches 80% capacity, `ConsolidationAgent` reads all raw entries and makes a per-entry decision using a structured LLM prompt: **promote** (new information), **compress** (merge with an existing LTM memory), or **discard** (noise). The agent also checks for contradictions across memory groups and persists `ContradictionFlag` records.

- **Long-term memory** — Promoted entries become `LTMMemory` objects stored in SQLite with full metadata and embedded in Qdrant using `nomic-embed-text` (768-dim). Queries are answered using semantic nearest-neighbour retrieval before the LLM sees the question, grounding responses in accumulated session knowledge.

---

## LHMBench — Long-Horizon Memory Benchmark

LHMBench is an open benchmark for evaluating long-horizon memory in coding assistants. Most existing evals test single-turn recall; LHMBench tests whether a system can **retain**, **filter**, **compress**, **detect contradictions in**, and **precisely retrieve** information across a multi-session conversation history.

### Scenarios

| Scenario | Description | What it tests |
|---|---|---|
| **Retention** | 5 facts ingested across 4 sessions; queried in order | Does the system remember facts from earlier sessions? |
| **Interference** | 1 signal fact + 5 noise facts + 1 misleading fact; query targets signal | Does signal survive interference from unrelated content? |
| **Contradiction** | Two conflicting user preferences ingested; system must detect or acknowledge | Is conflicting information flagged rather than silently accepted? |
| **Compression** | 5 detailed facts about one debugging session; queried after consolidation | Are key facts preserved through compression? |
| **Retrieval Precision** | 10 facts ingested (2 relevant); query targets the relevant subset | Does retrieval surface relevant facts without hallucinating from noise? |

### Example Results

| Scenario | engram | naive\_rag | no\_memory |
|---|---|---|---|
| Retention | **0.80** | 0.40 | 0.20 |
| Interference | **1.00** | 0.80 | 0.60 |
| Contradiction | **1.00** | 0.20 | 0.00 |
| Compression | **0.67** | 0.50 | 0.17 |
| Retrieval Precision | **1.00** | 0.70 | 0.30 |
| **Total** | **4.47 / 5.00 (89%)** | 2.60 / 5.00 (52%) | 1.27 / 5.00 (25%) |

*Results are illustrative; run `python -m benchmark.run_benchmark` for real numbers against your local Ollama instance.*

---

## Quickstart

### Prerequisites

- Python 3.11
- [Ollama](https://ollama.com/download) installed and running (`ollama serve`)

### Install

```bash
git clone https://github.com/vignesh-kumar-v/Engram.git
cd Engram
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Pull required models

```bash
ollama pull qwen2.5:7b          # main LLM
ollama pull nomic-embed-text    # embedding model for LTM
```

### Run the coding assistant

```bash
python -m hcma.scripts.run_agent
```

Available commands at the `>` prompt:

| Command | Effect |
|---|---|
| `status` | Show episodic buffer fill level |
| `history` | Show conversation turn count |
| `ltm` | Show LTM memory count and last 3 entries |
| `consolidate` | Manually trigger consolidation |
| `contradictions` | List unresolved contradiction flags |
| `quit` | Exit |

### Run the benchmark

```bash
python -m benchmark.run_benchmark
```

Run unit tests only (no Ollama required):

```bash
pytest hcma/tests/ benchmark/tests/ -m "not integration"
```

Run integration tests (requires Ollama):

```bash
pytest -m integration
```

---

## Project Structure

```
Engram/
├── hcma/                          # Core memory system
│   ├── agents/
│   │   ├── task_agent.py          # Ollama chat agent + episodic recording
│   │   └── consolidation_agent.py # LLM-driven promote/compress/discard
│   ├── memory/
│   │   ├── episodic_buffer.py     # SQLite-backed short-term buffer
│   │   ├── ltm_store.py           # SQLite + Qdrant long-term store
│   │   └── consolidation_loop.py  # Background daemon thread
│   ├── schemas/
│   │   └── memory_types.py        # EpisodicEntry, LTMMemory, Decision, …
│   ├── config/
│   │   └── settings.py            # All tuneable constants
│   ├── scripts/
│   │   └── run_agent.py           # Interactive CLI entry point
│   └── tests/                     # 177 unit tests
│
├── benchmark/                     # LHMBench evaluation harness
│   ├── scenarios/                 # retention, interference, contradiction, …
│   ├── systems/                   # engram, naive_rag, no_memory
│   ├── runner.py                  # BenchmarkRunner
│   ├── evaluator.py               # BenchmarkEvaluator + report table
│   ├── run_benchmark.py           # Entry point
│   ├── config.py                  # Benchmark constants
│   └── results/                   # JSON results (gitignored at runtime)
│
├── requirements.txt
├── pytest.ini
└── README.md
```

---

## Comparison with MemGPT

| | Engram | MemGPT |
|---|---|---|
| **Memory model** | Three-tier biological hierarchy (episodic → LTM) | Paged context with explicit memory functions |
| **Consolidation** | LLM reasons about each entry: promote / compress / discard | Storage operations triggered by context overflow |
| **Contradiction handling** | Dedicated detection + persistence of `ContradictionFlag` | Not addressed |
| **Benchmark** | Open LHMBench with 5 reproducible scenarios | No open long-horizon memory benchmark |
| **Infrastructure** | Local-first; SQLite + in-process Qdrant, no external services | Requires managed memory backend |
| **Scope** | Coding assistant domain; single-user sessions | General-purpose; multi-user production deployments |

Engram is a research prototype, not a production system. MemGPT is production-grade with a mature API. The honest tradeoff: Engram provides a more transparent, benchmarkable consolidation mechanism at the cost of scale and polish.

---

## Roadmap

- ✅ **Phase 1** — Core pipeline: EpisodicBuffer, LTMStore, TaskAgent, ConsolidationAgent
- ✅ **Phase 2** — Contradiction persistence, consolidation loop, CLI commands
- 🔄 **Phase 3** — LHMBench: scenario harness, BenchmarkRunner, BenchmarkEvaluator *(in progress)*
- ⬜ **Phase 4** — Multi-session evaluation: persistent sessions across process restarts, cross-session retention scoring
- ⬜ **Phase 5** — Fine-tuning: collect consolidation decisions from benchmark runs, fine-tune a smaller consolidation model on promote/compress/discard labels

---

## Citation

If you use Engram in your research, please cite:

```
Vigneshkumar Venugopal. Engram: Hierarchical Memory Consolidation for LLM Agents. 2026.
https://github.com/vignesh-kumar-v/Engram
```

---

*MIT License © 2026 Vigneshkumar Venugopal*
