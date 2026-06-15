"""
hybrid_retrieval/src/retrieval/hybrid_adapter.py
=================================================
Backend-facing adapter for the Hybrid Retrieval module.

Mirrors the pattern of vector_retrieval/src/retrieval/vector_adapter.py.
This is the REAL implementation. The backend bridge at
backend/app/retrieval/hybrid_adapter.py calls into this file via
sys.path injection — it never reimplements the logic itself.

Exposes two functions matching the shared adapter contract:

    ingest(chunks, document_id)  -> dict
    retrieve(query, top_k)       -> dict

Hybrid retrieval reads the shared indexes maintained by the other adapters:
    vector_retrieval/faiss_index.bin      — built by Collins's vector adapter
    vector_retrieval/chunk_records.npy    — built by Collins's vector adapter
    keyword_retrieval/keyword_bm25.pkl    — built by Olivier's keyword adapter
    keyword_retrieval/keyword_chunks.pkl  — built by Olivier's keyword adapter

RRF fusion uses Nathan's hybrid_retriever.reciprocal_rank_fusion().
BM25 tokenisation uses Nathan's preprocessing.tokenize_chunk().
Vector search uses Nathan's retrieval.vector_retriever.retrieve().
"""

import os
import pickle
import time
import threading
import logging

import numpy as np

logger = logging.getLogger(__name__)
_lock  = threading.RLock()

# ── Shared index paths (resolved from hybrid_retrieval/ root) ─────────────────
# The backend bridge os.chdir()s to hybrid_retrieval/ before calling us.
# We walk up one level to reach the project root, then into each module.
_HERE        = os.path.dirname(os.path.abspath(__file__))          # .../hybrid_retrieval/src/retrieval/
_MODULE_ROOT = os.path.abspath(os.path.join(_HERE, "../../.."))    # project root
_VEC_ROOT    = os.path.join(_MODULE_ROOT, "vector_retrieval")
_KW_ROOT     = os.path.join(_MODULE_ROOT, "keyword_retrieval")

_FAISS_PATH   = os.path.join(_VEC_ROOT, "faiss_index.bin")
_VECCHUNK_PATH= os.path.join(_VEC_ROOT, "chunk_records.npy")
_BM25_PATH    = os.path.join(_KW_ROOT,  "keyword_bm25.pkl")
_KWCHUNK_PATH = os.path.join(_KW_ROOT,  "keyword_chunks.pkl")

# ── In-memory cache ────────────────────────────────────────────────────────────
# retrieve() previously reloaded the SentenceTransformer model, the FAISS
# index, the vector chunk records, the BM25 model, and the keyword chunks
# from disk on EVERY call. SentenceTransformer load alone is the dominant
# cost (~1-3s), so a 10-question TriviaQA batch meant 10 redundant full
# reloads of all five artifacts. Cache them in memory; ingest() (in either
# the vector or keyword module) invalidates via _invalidate_cache().
_model_cache       = None  # SentenceTransformer
_faiss_index_cache = None  # faiss.Index
_vec_chunks_cache  = None  # list[dict]
_bm25_cache        = None  # BM25Okapi
_kw_chunks_cache   = None  # list[dict]


def _invalidate_cache():
    """Drop all cached models/indexes so the next retrieve() reloads from disk."""
    global _model_cache, _faiss_index_cache, _vec_chunks_cache, _bm25_cache, _kw_chunks_cache
    with _lock:
        _model_cache       = None
        _faiss_index_cache = None
        _vec_chunks_cache  = None
        _bm25_cache        = None
        _kw_chunks_cache   = None


def _get_model():
    """Load (and cache) the shared embedding model used for FAISS search."""
    global _model_cache
    with _lock:
        if _model_cache is None:
            from models.embedding_model import load_embedding_model
            _model_cache = load_embedding_model("all-MiniLM-L6-v2")
        return _model_cache


def _get_faiss_state():
    """Load (and cache) the FAISS index + vector chunk records."""
    global _faiss_index_cache, _vec_chunks_cache
    with _lock:
        if _faiss_index_cache is None or _vec_chunks_cache is None:
            import faiss
            _faiss_index_cache = faiss.read_index(_FAISS_PATH)
            _vec_chunks_cache  = np.load(_VECCHUNK_PATH, allow_pickle=True).tolist()
        return _faiss_index_cache, _vec_chunks_cache


def _get_bm25_state():
    """Load (and cache) the BM25 model + keyword chunk records."""
    global _bm25_cache, _kw_chunks_cache
    with _lock:
        if _bm25_cache is None or _kw_chunks_cache is None:
            with open(_BM25_PATH,    "rb") as f: _bm25_cache      = pickle.load(f)
            with open(_KWCHUNK_PATH, "rb") as f: _kw_chunks_cache = pickle.load(f)
        return _bm25_cache, _kw_chunks_cache


# ── Public interface ──────────────────────────────────────────────────────────

def ingest(chunks: list, document_id: str) -> dict:
    """
    Hybrid ingest: validates that both FAISS and BM25 indexes exist.

    No separate index is built — RRF fuses both indexes at query time.
    Both are maintained by vector_adapter and keyword_adapter respectively.

    Returns:
        {"status": "ok"|"error", "documents_ingested": int,
         "total_chunks": int, "latency_ms": float, "error": str|None}
    """
    start = time.perf_counter()
    with _lock:
        faiss_ok = os.path.exists(_FAISS_PATH)
        bm25_ok  = os.path.exists(_BM25_PATH)

        # The FAISS index and/or BM25 pickle on disk may have just been
        # rewritten by the vector/keyword adapters as part of this same
        # upload. Drop any cached copies so the next retrieve() reloads
        # the fresh versions rather than serving stale in-memory state.
        _invalidate_cache()

        latency_ms = round((time.perf_counter() - start) * 1000, 2)

        if faiss_ok and bm25_ok:
            print(
                f"  ✅ Hybrid: FAISS ✓  BM25 ✓  "
                f"({len(chunks)} chunks ready for RRF)"
            )
            return {
                "status":             "ok",
                "documents_ingested": 1,
                "total_chunks":       len(chunks),
                "latency_ms":         latency_ms,
                "error":              None,
            }

        missing = []
        if not faiss_ok: missing.append("faiss_index.bin")
        if not bm25_ok:  missing.append("keyword_bm25.pkl")
        msg = f"Hybrid ingest: waiting for {', '.join(missing)}"
        logger.warning(f"  ⚠️  {msg}")
        return {
            "status":             "error",
            "documents_ingested": 0,
            "total_chunks":       0,
            "latency_ms":         latency_ms,
            "error":              msg,
        }


def retrieve(query: str, top_k: int = 5) -> dict:
    """
    Hybrid retrieval: FAISS semantic + BM25 keyword, fused via RRF.

    Uses Nathan's own retriever implementations:
        - retrieval.vector_retriever.retrieve()    for FAISS
        - retrieval.bm25_retriever.retrieve_bm25() for BM25
        - retrieval.hybrid_retriever.reciprocal_rank_fusion() for RRF

    Returns:
        {"query": str, "method": "hybrid", "results": list[dict],
         "latency_ms": float}

    Raises:
        FileNotFoundError: if neither index exists yet
    """
    if not query or not query.strip():
        raise ValueError("query cannot be empty.")

    if not os.path.exists(_FAISS_PATH) and not os.path.exists(_BM25_PATH):
        raise FileNotFoundError(
            "No hybrid index found — upload documents first"
        )

    start = time.perf_counter()

    with _lock:
        try:
            from retrieval.vector_retriever  import retrieve as _vec_retrieve
            from retrieval.bm25_retriever    import retrieve_bm25
            from retrieval.hybrid_retriever  import reciprocal_rank_fusion
            from preprocessing.preprocess   import tokenize_chunk, detect_language

            # ── Step 1: FAISS vector retrieval ───────────────────────────
            vector_results = []
            if os.path.exists(_FAISS_PATH) and os.path.exists(_VECCHUNK_PATH):
                try:
                    index, chunks = _get_faiss_state()
                    model         = _get_model()
                    vector_results = _vec_retrieve(
                        query, model, index, chunks, top_k=top_k * 2
                    )
                except Exception as e:
                    logger.warning(f"  ⚠️  Hybrid FAISS sub-retrieval: {e}")

            # ── Step 2: BM25 keyword retrieval ───────────────────────────
            bm25_results = []
            if os.path.exists(_BM25_PATH) and os.path.exists(_KWCHUNK_PATH):
                try:
                    bm25, kw_chunks = _get_bm25_state()

                    _, nltk_lang = detect_language(query)
                    # Build a minimal inverted index for matched_terms reporting
                    stub_index   = {}
                    bm25_results = retrieve_bm25(
                        query, bm25, kw_chunks, stub_index,
                        nltk_lang=nltk_lang, top_k=top_k * 2
                    )
                except Exception as e:
                    logger.warning(f"  ⚠️  Hybrid BM25 sub-retrieval: {e}")

            # ── Step 3: RRF fusion ────────────────────────────────────────
            if vector_results or bm25_results:
                fused = reciprocal_rank_fusion(
                    bm25_results   = bm25_results,
                    vector_results = vector_results,
                    k              = 60,
                    top_k          = top_k,
                )
            else:
                fused = []

            # Normalise output to shared contract field names
            results = []
            for r in fused:
                src = r.get("source") or r.get("source_name") or "Unknown"
                results.append({
                    "rank":           r.get("rank",           len(results) + 1),
                    "chunk_id":       r.get("chunk_id",       ""),
                    "document_id":    r.get("document_id",    ""),
                    "document_title": r.get("document_title", ""),
                    "source":         src,
                    "source_name":    src,
                    "text":           r.get("text",           ""),
                    "score":          float(r.get("rrf_score") or r.get("score") or 0.0),
                    "rrf_score":      float(r.get("rrf_score") or 0.0),
                    "citation":       r.get("citation",       ""),
                    "metadata":       r.get("metadata",       {}),
                    "retrieval":      r.get("retrieval",      ""),
                })

            latency_ms = round((time.perf_counter() - start) * 1000, 2)
            return {
                "query":      query,
                "method":     "hybrid",
                "results":    results,
                "latency_ms": latency_ms,
            }

        except FileNotFoundError:
            raise
        except Exception as e:
            raise RuntimeError(f"Hybrid retrieval failed: {e}")
