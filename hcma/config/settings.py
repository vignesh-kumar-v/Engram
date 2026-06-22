"""HCMA global configuration settings."""

import os
from dotenv import load_dotenv

load_dotenv()

EPISODIC_BUFFER_CAPACITY: int = int(os.environ.get("EPISODIC_BUFFER_CAPACITY", "50"))
EPISODIC_IMPORTANCE_THRESHOLD: float = float(os.environ.get("EPISODIC_IMPORTANCE_THRESHOLD", "0.4"))
LTM_CONFIDENCE_THRESHOLD: float = float(os.environ.get("LTM_CONFIDENCE_THRESHOLD", "0.6"))

SQLITE_DB_PATH: str = os.environ.get("SQLITE_DB_PATH", "hcma/data/episodic.db")

QDRANT_HOST: str = os.environ.get("QDRANT_HOST", "localhost")
QDRANT_PORT: int = int(os.environ.get("QDRANT_PORT", "6333"))
LTM_COLLECTION_NAME: str = os.environ.get("LTM_COLLECTION_NAME", "hcma_ltm")

OLLAMA_BASE_URL: str = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL: str = os.environ.get("OLLAMA_MODEL", "qwen2.5:7b")

SESSION_ID_PREFIX: str = os.environ.get("SESSION_ID_PREFIX", "hcma_session")
CONSOLIDATION_MODEL: str = os.environ.get("CONSOLIDATION_MODEL", "qwen2.5:7b")
QDRANT_STORAGE_PATH: str = os.environ.get("QDRANT_STORAGE_PATH", "hcma/data/qdrant_storage")
CONSOLIDATION_CHECK_INTERVAL: int = int(os.environ.get("CONSOLIDATION_CHECK_INTERVAL", "30"))
CONSOLIDATION_TRIGGER_RATIO: float = float(os.environ.get("CONSOLIDATION_TRIGGER_RATIO", "0.8"))
LTM_TOP_K: int = int(os.environ.get("LTM_TOP_K", "3"))
