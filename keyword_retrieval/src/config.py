"""
config
==========
Central configuration for the Keyword Retrieval RAG pipeline.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Groq API ──────────────────────────────────────────────────────────────────
GROQ_API_KEY     = os.environ.get("GROQ_API_KEY", "")
QUERY_MODEL_NAME = "llama-3.1-8b-instant"
GENERATOR_MODEL  = "llama-3.1-8b-instant"
MAX_NEW_TOKENS   = 500
TEMPERATURE      = 0.1

# ── Storage ───────────────────────────────────────────────────────────────────
# Cross-platform: defaults to keyword_retrieval/tests/
# Override via KEYWORD_STORAGE_DIR env var
_HERE        = Path(__file__).resolve().parent.parent   # keyword_retrieval/
STORAGE_DIR  = os.environ.get(
    "KEYWORD_STORAGE_DIR",
    str(_HERE / "tests")
)

# ── Chunking defaults ─────────────────────────────────────────────────────────
DEFAULT_CHUNK_SIZE    = 400
DEFAULT_CHUNK_OVERLAP = 50

# ── Retrieval defaults ────────────────────────────────────────────────────────
DEFAULT_TOP_K = 5
