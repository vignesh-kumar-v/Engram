"""HCMA global configuration settings."""

EPISODIC_BUFFER_CAPACITY: int = 50
EPISODIC_IMPORTANCE_THRESHOLD: float = 0.4
LTM_CONFIDENCE_THRESHOLD: float = 0.6

SQLITE_DB_PATH: str = "hcma/data/episodic.db"

QDRANT_HOST: str = "localhost"
QDRANT_PORT: int = 6333
LTM_COLLECTION_NAME: str = "hcma_ltm"

OLLAMA_BASE_URL: str = "http://localhost:11434"
OLLAMA_MODEL: str = "qwen2.5:7b"

SESSION_ID_PREFIX: str = "hcma_session"
CONSOLIDATION_MODEL: str = "qwen2.5:7b"
QDRANT_STORAGE_PATH: str = "hcma/data/qdrant_storage"
CONSOLIDATION_CHECK_INTERVAL: int = 30
CONSOLIDATION_TRIGGER_RATIO: float = 0.8
