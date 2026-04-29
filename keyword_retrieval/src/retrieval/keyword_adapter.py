"""
src/retrieval/keyword_adapter.py
Backend-facing adapter for the Keyword Retrieval module.

This is the integration boundary between this module and the backend.
It exposes two clean functions that together form the complete plug-in interface:

    ingest(file_paths, chunk_size, chunk_overlap) -> dict
    retrieve(query, top_k) -> dict

The backend calls ingest() after a user uploads documents, then calls retrieve()
to answer queries against the indexed content. Neither function exposes any
internal implementation details (BM25, inverted index, chunker, etc).

The adapter manages all module-level state internally. The index is loaded once
on the first retrieve() call and reused across all subsequent calls for efficiency.
After ingest() rebuilds the index, the adapter reloads it automatically on the
next retrieve() call.

Contract references:
    shared_data/schemas/retrieval_request.schema.json
    shared_data/schemas/retrieval_response.schema.json
"""

import os
import time

from src.indexing.indexer    import build_pipeline
from src.indexing.bm25_store import load_bm25
from src.retrieval.retriever import retrieve as _retrieve


# ---------------------------------------------------------------------------
# persistence paths — configurable via environment variables
# ---------------------------------------------------------------------------
INDEX_PATH  = os.environ.get('KEYWORD_INDEX_PATH',  'keyword_index.pkl')
BM25_PATH   = os.environ.get('KEYWORD_BM25_PATH',   'keyword_bm25.pkl')
CHUNKS_PATH = os.environ.get('KEYWORD_CHUNKS_PATH', 'keyword_chunks.pkl')


# ---------------------------------------------------------------------------
# module-level state — loaded once, reused across calls
# ---------------------------------------------------------------------------
_bm25          = None
_index         = None
_chunk_records = None


def _load_state() -> None:
    """
    Load the BM25 model, inverted index, and chunk records into module-level state.
    Called lazily on first retrieve() call and after ingest() rebuilds the index.

    Raises:
        FileNotFoundError: If the index files do not exist (ingest not yet called).
    """
    global _bm25, _index, _chunk_records
    _bm25, _index, _chunk_records = load_bm25(BM25_PATH, INDEX_PATH, CHUNKS_PATH)


def _reset_state() -> None:
    """
    Clear index state so retrieve() reloads from the new index on next call.
    Called internally after ingest() rebuilds the index.
    """
    global _bm25, _index, _chunk_records
    _bm25          = None
    _index         = None
    _chunk_records = None


# ---------------------------------------------------------------------------
# public interface
# ---------------------------------------------------------------------------

def ingest(
    file_paths:    list,
    chunk_size:    int = 300,
    chunk_overlap: int = 50,
) -> dict:
    """
    Ingest one or more documents into the keyword index.

    Called by the backend after a user uploads files.
    Runs the full pipeline: load -> chunk -> tokenise -> inverted index + BM25 -> persist.
    After ingestion, retrieve() will search this new index.

    Args:
        file_paths:    List of absolute or relative file paths to ingest.
                       Supported types: .txt, .md, .pdf, .docx
        chunk_size:    Number of words per chunk. Default: 300.
        chunk_overlap: Number of overlapping words between adjacent chunks. Default: 50.

    Returns:
        {
            "status":             "ok" | "error",
            "documents_ingested": int,
            "total_chunks":       int,
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
        chunk_records, _ = build_pipeline(
            file_paths    = file_paths,
            chunk_size    = chunk_size,
            chunk_overlap = chunk_overlap,
            index_path    = INDEX_PATH,
            bm25_path     = BM25_PATH,
            chunks_path   = CHUNKS_PATH,
        )

        _reset_state()  # force retrieve() to reload the fresh index on next call

        latency_ms = round((time.perf_counter() - start) * 1000, 2)

        return {
            'status':             'ok',
            'documents_ingested': len(file_paths),
            'total_chunks':       len(chunk_records),
            'latency_ms':         latency_ms,
            'error':              None,
        }

    except Exception as e:
        latency_ms = round((time.perf_counter() - start) * 1000, 2)
        return {
            'status':             'error',
            'documents_ingested': 0,
            'total_chunks':       0,
            'latency_ms':         latency_ms,
            'error':              str(e),
        }


def retrieve(query: str, top_k: int = 3) -> dict:
    """
    Retrieve the top-k most relevant chunks for a query using BM25.

    Loads the index on first call; subsequent calls reuse state.
    The index searched is always the most recently ingested one.

    Args:
        query:  Natural language question string.
        top_k:  Number of top chunks to return. Default: 3.

    Returns:
        Dict matching retrieval_response.schema.json:
        {
            "query":      str,
            "method":     "keyword",
            "results":    list[dict],
            "latency_ms": float
        }

    Raises:
        ValueError:        If query is empty.
        FileNotFoundError: If index files are not found (ingest not yet called).
    """
    if not query or not query.strip():
        raise ValueError('query cannot be empty.')

    if _bm25 is None:
        _load_state()

    start      = time.perf_counter()
    results    = _retrieve(query, _bm25, _chunk_records, _index, top_k=top_k)
    latency_ms = round((time.perf_counter() - start) * 1000, 2)

    return {
        'query':      query,
        'method':     'keyword',
        'results':    results,
        'latency_ms': latency_ms,
    }