"""
src/retrieval/vector_adapter.py
Backend-facing adapter for the Vector Retrieval module.

This is the integration boundary between this module and the backend.
It exposes two clean functions that together form the complete plug-in interface:

    ingest(file_paths, document_ids, chunk_size, chunk_overlap) -> dict
    retrieve(query, top_k) -> dict

The backend calls ingest() after a user uploads documents, then calls retrieve()
to answer queries against the indexed content. Neither function exposes any
internal implementation details (FAISS, SentenceTransformer, chunker, etc).

The adapter manages all module-level state internally. The model is loaded once
on the first call and reused across all subsequent calls for efficiency.

Contract references:
    shared_data/schemas/retrieval_request.schema.json
    shared_data/schemas/retrieval_response.schema.json
"""

import os
import time
from typing import Any

from src.indexing.indexer import build_pipeline
from src.indexing.vector_store import load_index
from src.models.embedding_model import load_embedding_model
from src.retrieval.retriever import retrieve as _retrieve


# ---------------------------------------------------------------------------
# persistence paths — configurable via environment variables
# must match indexer.py defaults when not overridden
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
    Called lazily on first retrieve() call.

    Raises:
        FileNotFoundError: If the index or chunk records file does not exist.
        RuntimeError:      If the embedding model fails to load.
    """
    global _model, _index, _chunk_records

    if _model is None:
        _model = load_embedding_model(MODEL_NAME)

    if _index is None or _chunk_records is None:
        _index, _chunk_records = load_index(INDEX_PATH, CHUNKS_PATH)


def _reset_state() -> None:
    """
    Clear index state so retrieve() reloads from the new index on next call.
    Called internally after ingest() rebuilds the index.
    The model is preserved — no need to reload after re-indexing.
    """
    global _index, _chunk_records
    _index         = None
    _chunk_records = None


# ---------------------------------------------------------------------------
# public interface
# ---------------------------------------------------------------------------

def ingest(
    file_paths: list,
    chunk_size: int = 300,
    chunk_overlap: int = 50,
) -> dict:
    """
    Ingest one or more documents into the vector index.

    This is called by the backend after a user uploads files.
    It runs the full pipeline: load -> chunk -> embed -> FAISS index -> persist.
    After ingestion, retrieve() will search this new index.

    Args:
        file_paths:    List of absolute or relative file paths to ingest.
                       Supported types: .txt, .md, .pdf, .docx
        chunk_size:    Number of words per chunk. Default: 300.
        chunk_overlap: Number of overlapping words between adjacent chunks. Default: 50.

    Returns:
        Dict with ingestion summary:
        {
            "status":             "ok" | "error",
            "documents_ingested": int,
            "total_chunks":       int,
            "index_path":         str,
            "chunks_path":        str,
            "latency_ms":         float,
            "error":              str | None
        }

    Raises:
        ValueError: If file_paths is empty.
    """
    if not file_paths:
        raise ValueError('file_paths must contain at least one file path.')

    start = time.perf_counter()

    try:
        index, chunk_records, _ = build_pipeline(
            document_paths=file_paths,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            model_name=MODEL_NAME,
            index_path=INDEX_PATH,
            chunks_path=CHUNKS_PATH,
        )

        _reset_state()  # force retrieve() to reload the fresh index on next call

        latency_ms = round((time.perf_counter() - start) * 1000, 2)

        return {
            'status':             'ok',
            'documents_ingested': len(file_paths),
            'total_chunks':       len(chunk_records),
            'index_path':         INDEX_PATH,
            'chunks_path':        CHUNKS_PATH,
            'latency_ms':         latency_ms,
            'error':              None,
        }

    except Exception as e:
        latency_ms = round((time.perf_counter() - start) * 1000, 2)
        return {
            'status':             'error',
            'documents_ingested': 0,
            'total_chunks':       0,
            'index_path':         INDEX_PATH,
            'chunks_path':        CHUNKS_PATH,
            'latency_ms':         latency_ms,
            'error':              str(e),
        }


def retrieve(query: str, top_k: int = 3) -> dict:
    """
    Retrieve the top-k most relevant chunks for a query.

    Loads the model and index on first call; subsequent calls reuse state.
    The index searched is always the most recently ingested one.

    Args:
        query:  Natural language question string.
        top_k:  Number of top chunks to return. Default: 3.

    Returns:
        Dict matching retrieval_response.schema.json:
        {
            "query":      str,
            "method":     "vector",
            "results":    list[dict],
            "latency_ms": float
        }

    Raises:
        ValueError:        If query is empty.
        FileNotFoundError: If index files are not found at configured paths.
    """
    _load_state()

    start      = time.perf_counter()
    results    = _retrieve(query, _model, _index, _chunk_records, top_k=top_k)
    latency_ms = round((time.perf_counter() - start) * 1000, 2)

    return {
        'query':      query,
        'method':     'vector',
        'results':    results,
        'latency_ms': latency_ms,
    }
