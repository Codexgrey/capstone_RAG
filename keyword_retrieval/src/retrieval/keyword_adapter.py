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


# ── Public interface ──────────────────────────────────────────────────────────

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

        with open(_BM25_PATH,   "rb") as f: bm25   = pickle.load(f)
        with open(_CHUNKS_PATH, "rb") as f: chunks = pickle.load(f)

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
