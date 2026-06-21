"""Benchmark configuration."""

NUM_SESSIONS: int = 10
TURNS_PER_SESSION: int = 5
RETENTION_LAG: int = 3
TOP_K_RETRIEVAL: int = 7
OLLAMA_MODEL: str = "qwen2.5:7b"
EMBED_MODEL: str = "nomic-embed-text"
RESULTS_DIR: str = "benchmark/results"
SYSTEMS_TO_BENCHMARK: list = ["engram", "naive_rag", "no_memory"]
