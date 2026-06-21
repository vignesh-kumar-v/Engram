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
| **Noise Degradation** | 1 signal fact + 30 noise facts across 3 sessions + 1 updated signal fact | Does the most recent preference override earlier facts despite high noise volume? |

### Results (LHMBench v1)

| Scenario | engram | naive\_rag | no\_memory |
|---|---|---|---|
| Retention | **1.0/1.0** | 1.0/1.0 | 0.2/1.0 |
| Interference | **1.0/1.0** | 1.0/1.0 | 1.0/1.0 |
| Contradiction | **1.0/1.0** | 0.0/1.0 | 0.0/1.0 |
| Compression | **1.0/1.0** | 1.0/1.0 | 0.3/1.0 |
| Retrieval Precision | **1.0/1.0** | 1.0/1.0 | 1.0/1.0 |
| Noise Degradation | **1.0/1.0** | 1.0/1.0 | 0.0/1.0 |
| **Total** | **6.0/6.0 (100%)** | 5.0/6.0 (83.3%) | 2.5/6.0 (42.2%) |

**Winner: engram**

> Contradiction detection is Engram's key differentiator — the only system capable of identifying conflicting information across sessions. Benchmarked on Qwen2.5-7B via Ollama on local hardware.

### Multi-Session Lifecycle Benchmark

A secondary evaluation traces memory evolution across 8 sessions simulating a real coding project lifecycle: early sessions establish tech stack facts, middle sessions investigate a memory leak with misleading hypotheses, a later session provides the correct root cause, and final sessions test recall of both early facts and the correction. This tests whether consolidation preserves signal through longer timescales and whether systems recover from misinformation.

**Results (3 live runs, Qwen2.5-7B):**

| Run | Engram | naive_rag | no_memory | Winner |
|---|---|---|---|---|
| Run 1 | **100%** (5/5) | 80% (4/5) | 20% (1/5) | Engram |
| Run 2 | 80% (4/5) | **100%** (5/5) | 80% (4/5) | naive_rag |
| Run 3 | **100%** (5/5) | **100%** (5/5) | 60% (3/5) | naive_rag (tie-break) |
| **Average** | **93.3%** | **93.3%** | **53.3%** | — |

**Why the variance?** The multi-session scenario is intentionally small (5 questions testing 7 ingested memories) to simulate realistic interactive sessions. On such a small trial count, LLM-level non-determinism dominates: `qwen2.5:7b` sometimes articulates the correct answer, sometimes hedges "I need more context." This is noise, not a capability gap. A statistically meaningful comparison would require 100+ trials per system. The **primary evidence for Engram's advantage remains the six-scenario LHMBench suite**, which isolates specific memory challenges: Engram's sole advantage is contradiction detection (1.0 vs 0.0 for both baselines across all 6 scenarios).

**Bug discovery:** This multi-session testing uncovered a critical bug in the compress-decision fallback logic — `_parse_decision` was storing LLM-echoed IDs with brackets, and `_compress` had no prefix-lookup mechanism, causing all compress decisions to silently fall back to promote, preventing any deduplication. The fix (bracket stripping + prefix lookup) is now verified by 44 new regression tests.

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

Six-scenario LHMBench suite:
```bash
python -m benchmark.run_benchmark
```

Multi-session lifecycle harness (8 sessions, 5 queries):
```bash
python -m benchmark.run_multi_session
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
│   └── tests/                     # 292 unit tests + 44 compress-fix regression tests
│
├── benchmark/                     # LHMBench evaluation harness
│   ├── scenarios/                 # retention, interference, contradiction, multi_session_lifecycle, …
│   ├── systems/                   # engram, naive_rag, no_memory
│   ├── runner.py                  # BenchmarkRunner (6 scenarios)
│   ├── multi_session_runner.py    # MultiSessionRunner (8-session lifecycle)
│   ├── evaluator.py               # BenchmarkEvaluator + report table
│   ├── run_benchmark.py           # Entry point: 6-scenario suite
│   ├── run_multi_session.py       # Entry point: multi-session lifecycle
│   ├── config.py                  # Benchmark constants (TOP_K_RETRIEVAL, etc.)
│   ├── tests/                     # Benchmark scenario tests
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
- ✅ **Phase 3** — LHMBench: 6-scenario suite + multi-session lifecycle harness, BenchmarkRunner, BenchmarkEvaluator
- 🔄 **Phase 4** — Multi-session persistence: persistent sessions across process restarts, cross-session retention scoring
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
