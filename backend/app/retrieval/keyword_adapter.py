"""
backend/app/retrieval/keyword_adapter.py
=========================================
Thin bridge between Khalid's backend and Olivier's keyword_retrieval module.

All logic lives in:
    keyword_retrieval/src/retrieval/keyword_adapter.py

Uses importlib.util to load the module adapter directly by file path —
no __init__.py files required anywhere in keyword_retrieval/.

IMPORTANT: _KW_SRC is added to sys.path BEFORE exec_module and kept there
through the entire mod.ingest() / mod.retrieve() call. Olivier's module uses
lazy imports inside function bodies:
    from preprocessing.preprocess import tokenize_chunk, detect_language
These fire at call time, not at exec_module time, so sys.path must remain
intact for the full duration of each public function call.

CACHING
-------
Olivier's retrieve() re-unpickles keyword_bm25.pkl and keyword_chunks.pkl
from disk on every call (cheap relative to the FAISS+SentenceTransformer
cost in the vector/hybrid adapters, but still wasted work when repeated
across a TriviaQA batch — e.g. 10 redundant unpickles for a 10-question run).

Previously this bridge also called exec_module() on every retrieve(),
discarding any module-level state each time. The module is now loaded ONCE
and cached in _keyword_module so repeated calls reuse the same module
object. ingest() invalidates the cache so the next retrieve() picks up the
freshly rebuilt BM25 index/pickles.
"""

import sys
import os
import re
import threading
import importlib.util

_KW_ROOT    = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../../keyword_retrieval")
)
_KW_SRC     = os.path.join(_KW_ROOT, "src")
_KW_ADAPTER = os.path.join(_KW_SRC, "retrieval", "keyword_adapter.py")

# ── Module cache ──────────────────────────────────────────────────────────────
_keyword_module = None
_cache_lock      = threading.RLock()


def _load_keyword_module():
    if not os.path.exists(_KW_ADAPTER):
        raise FileNotFoundError(
            f"keyword_adapter.py not found at: {_KW_ADAPTER}\n"
            "Ensure keyword_retrieval/ is at the project root."
        )
    spec = importlib.util.spec_from_file_location(
        "keyword_retrieval.src.retrieval.keyword_adapter",
        _KW_ADAPTER,
        submodule_search_locations=[],
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _get_keyword_module():
    """Return the cached module, loading it on first use only."""
    global _keyword_module
    with _cache_lock:
        if _keyword_module is None:
            _keyword_module = _load_keyword_module()
        return _keyword_module


def _invalidate_cache():
    """Drop the cached module so the next call reloads the BM25 index from disk."""
    global _keyword_module
    with _cache_lock:
        _keyword_module = None


def ingest(chunks: list, document_id: str) -> dict:
    original_dir = os.getcwd()
    added = _KW_SRC not in sys.path
    if added:
        sys.path.insert(0, _KW_SRC)
    try:
        os.chdir(_KW_ROOT)
        mod = _get_keyword_module()
        return mod.ingest(chunks=chunks, document_id=document_id)
    except Exception as e:
        return {
            "status": "error", "documents_ingested": 0,
            "total_chunks": 0, "latency_ms": 0.0, "error": str(e),
        }
    finally:
        # Invalidate regardless of success/failure so a partially-rebuilt
        # BM25 pickle is re-read fresh on the next retrieve().
        _invalidate_cache()
        if added and _KW_SRC in sys.path:
            sys.path.remove(_KW_SRC)
        os.chdir(original_dir)


def retrieve(query: str, top_k: int = 5) -> list:
    original_dir = os.getcwd()
    added = _KW_SRC not in sys.path
    if added:
        sys.path.insert(0, _KW_SRC)
    try:
        os.chdir(_KW_ROOT)
        mod = _get_keyword_module()
        result = mod.retrieve(query=query, top_k=top_k)

        chunks = []
        for r in result.get("results", []):
            source = _clean(r.get("source") or r.get("source_name") or "Unknown")
            chunks.append({
                "chunk_id":       r.get("chunk_id",       ""),
                "document_id":    r.get("document_id",    ""),
                "document_title": r.get("document_title", ""),
                "source_name":    source,
                "source":         source,
                "text":           r.get("text",           ""),
                "score":          float(r.get("bm25_score") or r.get("score") or 0.0),
                "bm25_score":     float(r.get("bm25_score") or r.get("score") or 0.0),
                "rank":           r.get("rank", 0),
                "citation":       r.get("citation", ""),
                "metadata":       r.get("metadata", {}),
            })
        return chunks

    except FileNotFoundError:
        raise FileNotFoundError("No keyword index found — upload documents first")
    except Exception as e:
        _invalidate_cache()
        raise RuntimeError(f"Olivier keyword retrieval failed: {e}")
    finally:
        if added and _KW_SRC in sys.path:
            sys.path.remove(_KW_SRC)
        os.chdir(original_dir)


def _clean(s: str) -> str:
    return re.sub(r'^[0-9a-f]{8}_', '', s
                  .replace("_ocr.txt", ".pdf").replace("_ocr", ""))
