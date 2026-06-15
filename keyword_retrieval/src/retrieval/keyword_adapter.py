"""
keyword_retrieval/src/retrieval/keyword_adapter.py
===================================================
Backend-facing adapter for the Keyword Retrieval module.

Mirrors the pattern of vector_retrieval/src/retrieval/vector_adapter.py.
This is the REAL implementation. The backend bridge at
backend/app/retrieval/keyword_adapter.py calls into this file via
sys.path injection — it never reimplements the logic itself.

Exposes two functions matching the shared adapter contract:

    ingest(chunks, document_id)  -> dict
    retrieve(query, top_k)       -> dict

Index files saved to keyword_retrieval/ root (same folder as this module):
    keyword_bm25.pkl    — BM25Okapi model
    keyword_chunks.pkl  — list of chunk dicts
    keyword_index.pkl   — inverted index (stub, for future use)

The backend passes pre-parsed, pre-chunked chunk dicts (already in the
shared schema) so this adapter does NOT need to re-load or re-chunk files.
Tokenisation is done here using Olivier's preprocessing pipeline.
"""

import os
import pickle
import time
import threading
import logging

logger = logging.getLogger(__name__)

# ── Index paths (relative to keyword_retrieval/ root) ────────────────────────
# The backend bridge os.chdir()s to keyword_retrieval/ before calling us,
# so these relative paths resolve correctly in both contexts.
_BM25_PATH   = "keyword_bm25.pkl"
_CHUNKS_PATH = "keyword_chunks.pkl"
_INDEX_PATH  = "keyword_index.pkl"

_lock = threading.RLock()

# ── In-memory cache for the BM25 model + chunk records ────────────────────────
# retrieve() previously re-unpickled both files from disk on every call.
# That's cheap for a single query, but adds up across a TriviaQA batch
# (e.g. 10 redundant unpickles for a 10-question run). Cache them in memory
# and only reload when the cache is empty or has been explicitly invalidated
# by ingest().
_bm25_cache   = None
_chunks_cache = None


def _invalidate_cache():
    """Drop cached BM25 model + chunks so the next retrieve() reloads from disk."""
    global _bm25_cache, _chunks_cache
    with _lock:
        _bm25_cache   = None
        _chunks_cache = None


def _load_cached_state():
    """Load BM25 model + chunk records from disk, caching for subsequent calls."""
    global _bm25_cache, _chunks_cache
    with _lock:
        if _bm25_cache is None or _chunks_cache is None:
            with open(_BM25_PATH,   "rb") as f: _bm25_cache   = pickle.load(f)
            with open(_CHUNKS_PATH, "rb") as f: _chunks_cache = pickle.load(f)
        return _bm25_cache, _chunks_cache


# ── Public interface ──────────────────────────────────────────────────────────

def delete(document_id: str) -> dict:
    """
    Remove all chunks belonging to document_id from the BM25 index.

    Used for isolated per-question evaluation, where a document is
    ingested, queried, then removed before the next question starts —
    so no chunks carry over between questions.

    Rebuilds the BM25 model from the remaining chunks (same tokenisation
    pipeline as ingest()). If document_id is not present, or no index
    exists yet, this is a no-op and returns total_chunks for whatever
    is currently on disk (0 if nothing exists).

    Returns:
        {"status": "ok"|"error", "removed_chunks": int,
         "total_chunks": int, "error": str|None}
    """
    if not os.path.exists(_CHUNKS_PATH):
        return {"status": "ok", "removed_chunks": 0, "total_chunks": 0, "error": None}

    with _lock:
        try:
            with open(_CHUNKS_PATH, "rb") as f:
                existing = pickle.load(f)

            kept = [c for c in existing if c.get("document_id") != document_id]
            removed = len(existing) - len(kept)

            if removed == 0:
                return {"status": "ok", "removed_chunks": 0,
                        "total_chunks": len(existing), "error": None}

            if kept:
                from rank_bm25 import BM25Okapi
                from preprocessing.preprocess import tokenize_chunk, detect_language

                sample_text = kept[0]["text"]
                _, nltk_lang = detect_language(sample_text)
                tokenized = [tokenize_chunk(c["text"], nltk_lang) for c in kept]
                bm25 = BM25Okapi(tokenized)

                with open(_BM25_PATH, "wb") as f: pickle.dump(bm25, f)
                with open(_CHUNKS_PATH, "wb") as f: pickle.dump(kept, f)
            else:
                # Nothing left to index — remove the BM25 model and
                # chunk pickles entirely so retrieve() reports
                # "no index found" rather than scoring an empty corpus.
                if os.path.exists(_BM25_PATH):
                    os.remove(_BM25_PATH)
                if os.path.exists(_CHUNKS_PATH):
                    os.remove(_CHUNKS_PATH)

            _invalidate_cache()

            return {"status": "ok", "removed_chunks": removed,
                    "total_chunks": len(kept), "error": None}

        except Exception as e:
            logger.warning(f"  ⚠️  Keyword delete error: {e}")
            return {"status": "error", "removed_chunks": 0, "total_chunks": 0, "error": str(e)}


def ingest(chunks: list, document_id: str) -> dict:
    """
    Build/update the BM25 index with chunks from a newly uploaded document.

    Merges with any existing chunks, deduplicating by document_id so
    re-uploading the same document replaces only its prior chunks.

    The backend chunker already produces chunks in the shared schema:
        {chunk_id, document_id, source_name, text, page, metadata, ...}

    Tokenisation uses Olivier's preprocessing pipeline (NLTK + stemming)
    for consistency with the standalone keyword research pipeline.

    Returns:
        {"status": "ok"|"error", "documents_ingested": int,
         "total_chunks": int, "latency_ms": float, "error": str|None}
    """
    start = time.perf_counter()

    with _lock:
        try:
            from rank_bm25 import BM25Okapi
            from preprocessing.preprocess import tokenize_chunk, detect_language

            # Load existing state
            existing: list = []
            if os.path.exists(_CHUNKS_PATH):
                with open(_CHUNKS_PATH, "rb") as f:
                    existing = pickle.load(f)

            # Remove prior chunks for this document (re-ingest safety)
            existing = [c for c in existing if c.get("document_id") != document_id]

            all_chunks = existing + chunks

            # Detect language from first chunk for tokenisation
            sample_text = chunks[0]["text"] if chunks else ""
            _, nltk_lang = detect_language(sample_text)

            # Tokenise using Olivier's pipeline (lowercase → stopwords → stem)
            tokenized = [tokenize_chunk(c["text"], nltk_lang) for c in all_chunks]

            bm25 = BM25Okapi(tokenized)

            with open(_BM25_PATH,   "wb") as f: pickle.dump(bm25,      f)
            with open(_CHUNKS_PATH, "wb") as f: pickle.dump(all_chunks, f)
            # Write stub inverted index for compatibility
            if not os.path.exists(_INDEX_PATH):
                with open(_INDEX_PATH, "wb") as f: pickle.dump({}, f)

            # New BM25 model + chunks written to disk — drop the in-memory
            # cache so the next retrieve() picks up the rebuilt index.
            _invalidate_cache()

            latency_ms = round((time.perf_counter() - start) * 1000, 2)
            print(
                f"  ✅ Keyword (BM25): {len(chunks)} new, "
                f"{len(existing)} kept, {len(all_chunks)} total"
            )
            return {
                "status":             "ok",
                "documents_ingested": 1,
                "total_chunks":       len(all_chunks),
                "latency_ms":         latency_ms,
                "error":              None,
            }

        except Exception as e:
            latency_ms = round((time.perf_counter() - start) * 1000, 2)
            logger.warning(f"  ⚠️  Keyword ingest error: {e}")
            return {
                "status":             "error",
                "documents_ingested": 0,
                "total_chunks":       0,
                "latency_ms":         latency_ms,
                "error":              str(e),
            }


def retrieve(query: str, top_k: int = 5) -> dict:
    """
    Score all indexed chunks with BM25 and return the top_k results.

    Tokenises the query using the same NLTK pipeline as ingest()
    so query tokens match the index tokens exactly.

    Returns:
        {"query": str, "method": "keyword", "results": list[dict],
         "latency_ms": float}

    Raises:
        FileNotFoundError: if index has not been built yet (no uploads)
    """
    if not query or not query.strip():
        raise ValueError("query cannot be empty.")

    if not os.path.exists(_BM25_PATH) or not os.path.exists(_CHUNKS_PATH):
        raise FileNotFoundError(
            "No keyword index found — upload documents first"
        )

    start = time.perf_counter()

    try:
        from preprocessing.preprocess import tokenize_chunk, detect_language

        bm25, chunks = _load_cached_state()

        _, nltk_lang    = detect_language(query)
        query_tokens    = tokenize_chunk(query, nltk_lang)

        if not query_tokens:
            # All words were stopwords — fall back to simple whitespace split
            query_tokens = query.lower().split()

        scores  = bm25.get_scores(query_tokens)
        safe_k  = min(top_k, len(chunks))
        top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:safe_k]

        results = []
        for rank, idx in enumerate(top_idx):
            c = chunks[idx]
            results.append({
                "rank":           rank + 1,
                "chunk_id":       c.get("chunk_id",       f"bm25_{idx}"),
                "document_id":    c.get("document_id",    ""),
                "document_title": c.get("document_title", ""),
                "source":         c.get("source_name") or c.get("source") or "Unknown",
                "source_name":    c.get("source_name") or c.get("source") or "Unknown",
                "text":           c.get("text",           ""),
                "score":          float(scores[idx]),
                "bm25_score":     float(scores[idx]),
                "citation":       c.get("citation", ""),
                "metadata":       c.get("metadata", {}),
                "page":           c.get("page"),
            })

        latency_ms = round((time.perf_counter() - start) * 1000, 2)
        return {
            "query":      query,
            "method":     "keyword",
            "results":    results,
            "latency_ms": latency_ms,
        }

    except FileNotFoundError:
        raise
    except Exception as e:
        raise RuntimeError(f"Keyword retrieval failed: {e}")
