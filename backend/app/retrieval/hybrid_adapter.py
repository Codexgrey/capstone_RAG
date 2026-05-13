"""
backend/app/retrieval/hybrid_adapter.py
========================================
Thin bridge between Khalid's backend and Nathan's hybrid_retrieval module.

All logic lives in:
    hybrid_retrieval/src/retrieval/hybrid_adapter.py

Uses importlib.util to load the module adapter directly by file path —
no __init__.py files required anywhere in hybrid_retrieval/.

IMPORTANT: _HY_SRC is added to sys.path BEFORE exec_module and kept there
through the entire mod.ingest() / mod.retrieve() call. Nathan's module uses
lazy imports inside function bodies:
    from retrieval.vector_retriever import ...
    from preprocessing.preprocess import ...
    from models.embedding_model import ...
These fire at call time, not at exec_module time, so sys.path must remain
intact for the full duration of each public function call.
"""

import sys
import os
import re
import importlib.util

_HY_ROOT    = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../../hybrid_retrieval")
)
_HY_SRC     = os.path.join(_HY_ROOT, "src")
_HY_ADAPTER = os.path.join(_HY_SRC, "retrieval", "hybrid_adapter.py")


def _load_hybrid_module():
    if not os.path.exists(_HY_ADAPTER):
        raise FileNotFoundError(
            f"hybrid_adapter.py not found at: {_HY_ADAPTER}\n"
            "Ensure hybrid_retrieval/ is at the project root."
        )
    spec = importlib.util.spec_from_file_location(
        "hybrid_retrieval.src.retrieval.hybrid_adapter",
        _HY_ADAPTER,
        submodule_search_locations=[],
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def ingest(chunks: list, document_id: str) -> dict:
    original_dir = os.getcwd()
    added = _HY_SRC not in sys.path
    if added:
        sys.path.insert(0, _HY_SRC)
    try:
        os.chdir(_HY_ROOT)
        mod = _load_hybrid_module()
        return mod.ingest(chunks=chunks, document_id=document_id)
    except Exception as e:
        return {
            "status": "error", "documents_ingested": 0,
            "total_chunks": 0, "latency_ms": 0.0, "error": str(e),
        }
    finally:
        if added and _HY_SRC in sys.path:
            sys.path.remove(_HY_SRC)
        os.chdir(original_dir)


def retrieve(query: str, top_k: int = 5) -> list:
    original_dir = os.getcwd()
    added = _HY_SRC not in sys.path
    if added:
        sys.path.insert(0, _HY_SRC)
    try:
        os.chdir(_HY_ROOT)
        mod = _load_hybrid_module()
        result = mod.retrieve(query=query, top_k=top_k)

        chunks = []
        for r in result.get("results", []):
            source = _clean(r.get("source") or r.get("source_name") or "Unknown")
            chunks.append({
                "chunk_id":         r.get("chunk_id",       ""),
                "document_id":      r.get("document_id",    ""),
                "document_title":   r.get("document_title", ""),
                "source_name":      source,
                "source":           source,
                "text":             r.get("text",           ""),
                "score":            float(r.get("rrf_score") or r.get("score") or 0.0),
                "rrf_score":        float(r.get("rrf_score") or r.get("score") or 0.0),
                "rank":             r.get("rank", 0),
                "citation":         r.get("citation", ""),
                "metadata":         r.get("metadata", {}),
                "retrieval_source": r.get("retrieval", ""),
            })
        return chunks

    except FileNotFoundError:
        raise FileNotFoundError("No hybrid index found — upload documents first")
    except Exception as e:
        raise RuntimeError(f"Nathan hybrid retrieval failed: {e}")
    finally:
        if added and _HY_SRC in sys.path:
            sys.path.remove(_HY_SRC)
        os.chdir(original_dir)


def _clean(s: str) -> str:
    return re.sub(r'^[0-9a-f]{8}_', '', s
                  .replace("_ocr.txt", ".pdf").replace("_ocr", ""))
