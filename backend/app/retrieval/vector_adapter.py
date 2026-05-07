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
"""

import sys
import os
import re
import importlib.util

_VECTOR_ROOT    = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../../vector_retrieval")
)
_VECTOR_SRC     = os.path.join(_VECTOR_ROOT, "src")
_VECTOR_ADAPTER = os.path.join(_VECTOR_SRC, "retrieval", "vector_adapter.py")


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


def ingest(file_paths: list, chunk_size: int = 300, chunk_overlap: int = 50) -> dict:
    original_dir = os.getcwd()
    added_root = _VECTOR_ROOT not in sys.path
    if added_root:
        sys.path.insert(0, _VECTOR_ROOT)
    try:
        os.chdir(_VECTOR_ROOT)
        mod = _load_vector_module()
        return mod.ingest(
            file_paths=file_paths,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
    except Exception as e:
        return {"status": "error", "error": str(e),
                "documents_ingested": 0, "total_chunks": 0}
    finally:
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
        mod = _load_vector_module()
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
        raise RuntimeError(f"Collins vector retrieval failed: {e}")
    finally:
        if added_root and _VECTOR_ROOT in sys.path:
            sys.path.remove(_VECTOR_ROOT)
        os.chdir(original_dir)
