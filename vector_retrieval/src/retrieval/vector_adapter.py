"""
src/retrieval/vector_adapter.py
Backend-facing adapter for the Vector Retrieval module.

This is the integration boundary between this module and the backend.
It exposes a single clean function that matches the shared retrieval contract:

    retrieve(query, top_k) -> dict

The backend adapter layer (backend/app/retrieval/vector/) calls this function.
It does not need to know about FAISS, SentenceTransformer, or chunk records —
the adapter handles loading and state internally.

Contract reference: shared_data/schemas/retrieval_response.schema.json
"""

import time
import os
from src.indexing.vector_store import load_index
from src.models.embedding_model import load_embedding_model
from src.retrieval.retriever import retrieve as _retrieve
from typing import Any

# ---------------------------------------------------------------------------
# persistence paths — must match indexer.py defaults
# ---------------------------------------------------------------------------
INDEX_PATH  = os.environ.get('VECTOR_INDEX_PATH',  'faiss_index.bin')
CHUNKS_PATH = os.environ.get('VECTOR_CHUNKS_PATH', 'chunk_records.npy')
MODEL_NAME  = os.environ.get('VECTOR_MODEL_NAME',  'all-MiniLM-L6-v2')


# ---------------------------------------------------------------------------
# module-level state — loaded once, reused across calls
# ---------------------------------------------------------------------------
_model         = None
_index         = None
_chunk_records = None


def _load_state() -> None:
    """
    Load the embedding model and FAISS index into module-level state.
    Called once on first retrieve() call (lazy initialisation).

    Raises:
        FileNotFoundError: If the index or chunk records file does not exist.
        RuntimeError:      If the embedding model fails to load.
    """
    global _model, _index, _chunk_records

    if _model is None:
        _model = load_embedding_model(MODEL_NAME)

    if _index is None or _chunk_records is None:
        _index, _chunk_records = load_index(INDEX_PATH, CHUNKS_PATH)


def retrieve(query: str, top_k: int = 3) -> dict[str, Any]:
    """
    Retrieve the top-k most relevant chunks for a query.

    This is the public interface for backend integration.
    Loads the model and index on first call; subsequent calls reuse state.

    Args:
        query:  Natural language question string.
        top_k:  Number of top chunks to return. Default: 3.

    Returns:
        Dict matching retrieval_response.schema.json:
        {
            "query":      str,
            "method":     "vector",
            "results":    list[dict],   # top-k chunks, ordered by score desc
            "latency_ms": float
        }

    Raises:
        ValueError:       If query is empty.
        FileNotFoundError: If index files are not found at configured paths.
    """
    _load_state()

    start = time.perf_counter()
    results = _retrieve(query, _model, _index, _chunk_records, top_k=top_k)
    latency_ms = (time.perf_counter() - start) * 1000

    return {
        'query':      query,
        'method':     'vector',
        'results':    results,
        'latency_ms': round(latency_ms, 2),
    }
