"""
backend/app/retrieval/vector_adapter.py
========================================
Thin bridge between Khalid's backend and Collins's vector_retrieval module.

All logic lives in:
    vector_retrieval/src/retrieval/vector_adapter.py

Uses importlib.util to load the module adapter directly by file path —
consistent with keyword_adapter.py and hybrid_adapter.py.

Collins's module uses module-level imports (from src.utils.loader etc.),
so _VECTOR_ROOT must be on sys.path during exec_module AND through the call.
It is added before exec_module and removed only after ingest()/retrieve()
complete, matching the pattern used by the keyword and hybrid bridges.

SCHEMA-AGNOSTIC INGESTION
-------------------------
ingest() now takes (chunks, document_id) — the same shared-schema chunk
dicts produced by backend/app/ingestion/chunker.py (fixed 200-word /
40-word-overlap chunks) and already consumed by keyword_adapter.ingest()
and hybrid_adapter.ingest(). Vector no longer re-chunks raw files via its
own module chunker; this guarantees Vector, Keyword, and Hybrid all index
the same chunk boundaries, matching the shared adapter contract.

CACHING
-------
Collins's module keeps the loaded SentenceTransformer model, the FAISS
index, and the chunk records as module-level globals (_model, _index,
_chunk_records), reloading them lazily on first retrieve(). The module is
loaded ONCE and cached in _vector_module. Subsequent retrieve() calls reuse
the same module object, so Collins's own module-level caching
(_model/_index/_chunk_records) actually takes effect. ingest() invalidates
the cache by clearing _vector_module so the next retrieve() picks up a
fresh module with a freshly-loaded index that reflects the newly ingested
documents.
"""

import sys
import os
import re
import threading
import importlib.util

_VECTOR_ROOT    = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../../vector_retrieval")
)
_VECTOR_SRC     = os.path.join(_VECTOR_ROOT, "src")
_VECTOR_ADAPTER = os.path.join(_VECTOR_SRC, "retrieval", "vector_adapter.py")

# ── Module cache ──────────────────────────────────────────────────────────────
_vector_module = None
_cache_lock     = threading.RLock()


def _load_vector_module():
    if not os.path.exists(_VECTOR_ADAPTER):
        raise FileNotFoundError(
            f"vector_adapter.py not found at: {_VECTOR_ADAPTER}\n"
            "Ensure vector_retrieval/ is at the project root."
        )
    spec = importlib.util.spec_from_file_location(
        "vector_retrieval.src.retrieval.vector_adapter",
        _VECTOR_ADAPTER,
        submodule_search_locations=[],
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _get_vector_module():
    """Return the cached module, loading it on first use only."""
    global _vector_module
    with _cache_lock:
        if _vector_module is None:
            _vector_module = _load_vector_module()
        return _vector_module


def _invalidate_cache():
    """Drop the cached module so the next call reloads model + index from disk."""
    global _vector_module
    with _cache_lock:
        _vector_module = None


def ingest(chunks: list, document_id: str) -> dict:
    original_dir = os.getcwd()
    added_root = _VECTOR_ROOT not in sys.path
    if added_root:
        sys.path.insert(0, _VECTOR_ROOT)
    try:
        os.chdir(_VECTOR_ROOT)
        mod = _get_vector_module()
        return mod.ingest(chunks=chunks, document_id=document_id)
    except Exception as e:
        return {"status": "error", "error": str(e),
                "documents_ingested": 0, "total_chunks": 0}
    finally:
        # Invalidate regardless of success/failure — if ingest partially
        # wrote a new index, we want the next retrieve() to reload it
        # rather than serve a stale cached index/chunk_records mismatch.
        _invalidate_cache()
        if added_root and _VECTOR_ROOT in sys.path:
            sys.path.remove(_VECTOR_ROOT)
        os.chdir(original_dir)


def retrieve(query: str, top_k: int = 5) -> list:
    original_dir = os.getcwd()
    added_root = _VECTOR_ROOT not in sys.path
    if added_root:
        sys.path.insert(0, _VECTOR_ROOT)
    try:
        os.chdir(_VECTOR_ROOT)
        mod = _get_vector_module()
        result = mod.retrieve(query=query, top_k=top_k)

        chunks = []
        for r in result.get("results", []):
            source = r.get("source", "")
            if source.endswith("_ocr.txt"):
                source = source.replace("_ocr.txt", ".pdf")
            source = re.sub(r'^[0-9a-f]{8}_', '', source)

            chunks.append({
                "chunk_id":       r.get("chunk_id", ""),
                "document_id":    r.get("document_id", ""),
                "source_name":    source,
                "text":           r.get("text", ""),
                "score":          r.get("score", 0.0),
                "rank":           r.get("rank", 0),
                "document_title": r.get("document_title", "").replace("_ocr", "").replace("Ocr", "").strip(),
                "source":         source,
                "citation":       r.get("citation", ""),
                "metadata":       r.get("metadata", {}),
            })
        return chunks

    except FileNotFoundError:
        raise FileNotFoundError("No FAISS index found — upload documents first")
    except Exception as e:
        # A stale cached module could be the cause of an unexpected error
        # (e.g. index file was deleted externally) — drop it so the next
        # call gets a clean reload instead of repeating the same failure.
        _invalidate_cache()
        raise RuntimeError(f"Collins vector retrieval failed: {e}")
    finally:
        if added_root and _VECTOR_ROOT in sys.path:
            sys.path.remove(_VECTOR_ROOT)
        os.chdir(original_dir)
