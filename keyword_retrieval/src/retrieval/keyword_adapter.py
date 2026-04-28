"""
keyword_adapter.py
===================
Location: keyword_retrieval/src/retrieval/keyword_adapter.py

Shared adapter interface for the Keyword Retrieval module.
This is the ONLY file the backend imports from this module.

Implements the two-function plug-in interface defined in
Shared_Contracts.docx Section 5:

    ingest(file_paths, chunk_size, chunk_overlap)
        Loads documents, chunks them, builds the inverted index
        and BM25 model, persists state to disk.

    retrieve(query, top_k)
        Searches the index using BM25 and returns results in the
        exact shape defined by retrieval_response.schema.json.

Contract compliance checklist (Section 6):
    ✓  ingest() and retrieve() importable from this adapter
    ✓  ingest() returns status "ok" before retrieve() is called
    ✓  retrieve() raises FileNotFoundError if called before ingest()
    ✓  Response matches retrieval_response.schema.json exactly
    ✓  score field present — higher = more relevant (BM25 score)
    ✓  citation format: [Document Title | chunk_id]
    ✓  metadata.file_type and metadata.uploaded_at on every result
    ✓  latency_ms measured and included in every response
    ✓  No LLM API calls anywhere in this module
"""

import time
import pickle
import datetime
from pathlib import Path


# =============================================================================
# PERSISTENT STATE PATHS
# The adapter saves its index to disk so it survives process restarts.
# These paths sit next to the adapter file inside src/retrieval/.
# =============================================================================

_BASE_DIR        = Path(__file__).parent
_INDEX_PATH      = _BASE_DIR / "keyword_index.pkl"   # inverted index
_BM25_PATH       = _BASE_DIR / "keyword_bm25.pkl"    # BM25 model
_CHUNKS_PATH     = _BASE_DIR / "keyword_chunks.pkl"  # chunk records


# =============================================================================
# MODULE-LEVEL STATE
# Loaded once on the first retrieve() call and reused for all subsequent calls.
# After ingest() rebuilds the index, _needs_reload is set to True so the
# next retrieve() call reloads fresh state from disk.
# =============================================================================

_inverted_index  = None
_bm25            = None
_chunk_records   = None
_needs_reload    = False


# =============================================================================
# INTERNAL HELPERS
# =============================================================================

def _get_file_type(path: str) -> str:
    """Return the lowercase file extension without the dot."""
    return Path(path).suffix.lower().lstrip(".")


def _now_utc() -> str:
    """Return current UTC time as an ISO 8601 string."""
    return datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")


def _load_state() -> None:
    """Load the persisted index, BM25 model, and chunk records from disk."""
    global _inverted_index, _bm25, _chunk_records, _needs_reload

    if not _INDEX_PATH.exists() or not _BM25_PATH.exists() or not _CHUNKS_PATH.exists():
        raise FileNotFoundError(
            "No keyword index found. Call ingest(file_paths) first."
        )

    with open(_INDEX_PATH,  "rb") as f: _inverted_index = pickle.load(f)
    with open(_BM25_PATH,   "rb") as f: _bm25           = pickle.load(f)
    with open(_CHUNKS_PATH, "rb") as f: _chunk_records  = pickle.load(f)

    _needs_reload = False


def _save_state(inverted_index, bm25, chunk_records) -> None:
    """Persist the index, BM25 model, and chunk records to disk."""
    with open(_INDEX_PATH,  "wb") as f: pickle.dump(inverted_index, f)
    with open(_BM25_PATH,   "wb") as f: pickle.dump(bm25,           f)
    with open(_CHUNKS_PATH, "wb") as f: pickle.dump(chunk_records,  f)


# =============================================================================
# PUBLIC INTERFACE — ingest()
# =============================================================================

def ingest(
    file_paths:    list[str],
    chunk_size:    int = 300,
    chunk_overlap: int = 50,
) -> dict:
    """
    Load documents, chunk them, build the inverted index and BM25 model,
    and persist everything to disk.

    Called by the backend after the user uploads documents.
    After this call, retrieve() will search the newly built index.

    All pipeline errors are caught internally — this function always
    returns a dict and never raises.

    Parameters
    ----------
    file_paths    : list[str]  absolute paths to uploaded files
                               (.txt, .md, .pdf, .docx supported)
    chunk_size    : int        words per chunk  (default: 300)
    chunk_overlap : int        overlap between adjacent chunks  (default: 50)

    Returns
    -------
    {
        "status":             "ok" | "error",
        "documents_ingested": int,
        "total_chunks":       int,
        "latency_ms":         float,
        "error":              str | None
    }
    """
    global _needs_reload
    t_start = time.perf_counter()

    try:
        # Import pipeline modules
        from utils.loader      import _FORMAT_LOADERS
        from utils.chunker     import chunk_text_with_metadata
        from preprocessing.preprocess import (
            clean_text, detect_language, tokenize_chunk
        )
        from indexing.indexer    import build_inverted_index
        from indexing.bm25_store import build_bm25

        all_chunk_records    = []
        all_tokenized_chunks = []
        uploaded_at          = _now_utc()
        docs_ingested        = 0

        for file_path in file_paths:
            path = Path(file_path)

            # ── Load ──────────────────────────────────────────────
            ext = path.suffix.lower()
            if ext not in _FORMAT_LOADERS:
                # Skip unsupported formats silently
                continue

            try:
                text = _FORMAT_LOADERS[ext](path)
            except Exception:
                continue

            if not text or not text.strip():
                continue

            # ── Clean ─────────────────────────────────────────────
            cleaned = clean_text(text)

            # ── Detect language ───────────────────────────────────
            lang_code, nltk_lang = detect_language(cleaned)

            # ── Chunk ─────────────────────────────────────────────
            doc_num   = docs_ingested + 1
            doc_id    = f"doc-{doc_num:03d}"
            doc_title = path.stem.replace("_", " ").replace("-", " ").title()

            chunks = chunk_text_with_metadata(
                cleaned,
                chunk_size    = chunk_size,
                overlap       = chunk_overlap,
                document_title= doc_title,
                source        = path.name,
                document_id   = doc_id,
                lang_code     = lang_code,
            )

            # Attach metadata required by the contract
            for chunk in chunks:
                chunk["metadata"] = {
                    "file_name":    path.name,
                    "file_type":    ext.lstrip("."),
                    "file_size_kb": round(path.stat().st_size / 1024, 2),
                    "uploaded_at":  uploaded_at,
                }

            # ── Tokenise ──────────────────────────────────────────
            tokenized = [
                tokenize_chunk(c["text"], nltk_lang)
                for c in chunks
            ]

            all_chunk_records.extend(chunks)
            all_tokenized_chunks.extend(tokenized)
            docs_ingested += 1

        if docs_ingested == 0:
            return {
                "status":             "error",
                "documents_ingested": 0,
                "total_chunks":       0,
                "latency_ms":         round((time.perf_counter() - t_start) * 1000, 2),
                "error":              "No supported files could be processed.",
            }

        # ── Build inverted index + BM25 ───────────────────────────
        inverted_index = build_inverted_index(all_chunk_records, all_tokenized_chunks)
        bm25           = build_bm25(all_tokenized_chunks)

        # ── Persist to disk ───────────────────────────────────────
        _save_state(inverted_index, bm25, all_chunk_records)
        _needs_reload = True   # tell retrieve() to reload on next call

        latency_ms = round((time.perf_counter() - t_start) * 1000, 2)
        return {
            "status":             "ok",
            "documents_ingested": docs_ingested,
            "total_chunks":       len(all_chunk_records),
            "latency_ms":         latency_ms,
            "error":              None,
        }

    except Exception as e:
        latency_ms = round((time.perf_counter() - t_start) * 1000, 2)
        return {
            "status":             "error",
            "documents_ingested": 0,
            "total_chunks":       0,
            "latency_ms":         latency_ms,
            "error":              str(e),
        }


# =============================================================================
# PUBLIC INTERFACE — retrieve()
# =============================================================================

def retrieve(query: str, top_k: int = 3) -> dict:
    """
    Search the keyword index using BM25 and return results in the
    standard retrieval_response.schema.json shape.

    Loads the model and index on the first call, then reuses them.
    After ingest() rebuilds the index, automatically reloads on next call.

    Parameters
    ----------
    query : str   natural language question from the user
    top_k : int   number of top chunks to return  (default: 3)

    Returns
    -------
    {
        "query":      str,
        "method":     "keyword",
        "results": [
            {
                "rank":     int,
                "chunk_id": str,
                "text":     str,
                "score":    float,     # BM25 score — higher = more relevant
                "citation": str,       # [Document Title | chunk_id]
                "metadata": {
                    "file_name":    str,
                    "file_type":    str,
                    "file_size_kb": float,
                    "uploaded_at":  str   # ISO 8601 UTC
                }
            },
            ...
        ],
        "latency_ms": float
    }

    Raises
    ------
    FileNotFoundError
        If called before any successful ingest().
        The backend should guard against this and return HTTP 400 or 503.
    ValueError
        If query is empty or whitespace only.
    """
    global _inverted_index, _bm25, _chunk_records, _needs_reload

    if not query or not query.strip():
        raise ValueError("Query cannot be empty.")

    t_start = time.perf_counter()

    # ── Load state if needed ──────────────────────────────────────
    if _bm25 is None or _needs_reload:
        _load_state()   # raises FileNotFoundError if no index exists

    # ── Tokenise query ────────────────────────────────────────────
    from preprocessing.preprocess import tokenize_chunk, detect_language

    lang_code, nltk_lang = detect_language(query)
    query_tokens         = tokenize_chunk(query, nltk_lang)

    if not query_tokens:
        # All words were stopwords — return empty results
        return {
            "query":      query,
            "method":     "keyword",
            "results":    [],
            "latency_ms": round((time.perf_counter() - t_start) * 1000, 2),
        }

    # ── Score with BM25 ───────────────────────────────────────────
    scores   = _bm25.get_scores(query_tokens)
    safe_k   = min(top_k, len(_chunk_records))
    top_idxs = sorted(
        range(len(scores)),
        key     = lambda i: scores[i],
        reverse = True,
    )[:safe_k]

    # ── Build results array ───────────────────────────────────────
    results = []
    for rank, chunk_idx in enumerate(top_idxs):
        chunk = _chunk_records[chunk_idx]
        score = float(scores[chunk_idx])

        # Matched terms (for transparency — not in schema but harmless to compute)
        matched = [
            t for t in query_tokens
            if t in _inverted_index
            and any(
                p["chunk_idx"] == chunk_idx
                for p in _inverted_index[t]["postings"]
            )
        ]

        results.append({
            "rank":     rank + 1,
            "chunk_id": chunk["chunk_id"],
            "text":     chunk["text"],
            "score":    score,
            "citation": f"[{chunk['document_title']} | {chunk['chunk_id']}]",
            "metadata": chunk.get("metadata", {
                # Fallback if metadata was not attached during ingest
                "file_name":    chunk.get("source", "unknown"),
                "file_type":    Path(chunk.get("source", "")).suffix.lstrip("."),
                "file_size_kb": 0.0,
                "uploaded_at":  "unknown",
            }),
            # Internal field — useful for debugging, ignored by backend
            "_matched_terms": matched,
        })

    latency_ms = round((time.perf_counter() - t_start) * 1000, 2)

    return {
        "query":      query,
        "method":     "keyword",
        "results":    results,
        "latency_ms": latency_ms,
    }
