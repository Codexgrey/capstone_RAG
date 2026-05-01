"""
app/retrieval/vector_adapter.py
Bridge between Khalid's backend and Collins's vector_retrieval module.

FIX: sys.path injection moved INSIDE functions so it doesn't pollute the
global sys.path at import time and cause namespace collisions with Olivier's
and Nathan's modules.
"""

import sys
import os
import re

_VECTOR_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../../vector_retrieval")
)
_COLLINS_SRC = os.path.join(_VECTOR_ROOT, "src")

# ── NOT at module level anymore — moved inside each function 


def ingest(file_paths: list, chunk_size: int = 300, chunk_overlap: int = 50) -> dict:
    original_dir = os.getcwd()
    try:
        # Inject Collins's paths only while we need them
        for p in [_VECTOR_ROOT, _COLLINS_SRC]:
            if p not in sys.path:
                sys.path.insert(0, p)

        os.chdir(_VECTOR_ROOT)

        from retrieval.vector_adapter import ingest as collins_ingest
        return collins_ingest(
            file_paths=file_paths,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
    except Exception as e:
        return {"status": "error", "error": str(e),
                "documents_ingested": 0, "total_chunks": 0}
    finally:
        # Remove Collins's paths so they don't linger for other modules
        for p in [_VECTOR_ROOT, _COLLINS_SRC]:
            if p in sys.path:
                sys.path.remove(p)
        os.chdir(original_dir)


def retrieve(query: str, top_k: int = 5) -> list:
    original_dir = os.getcwd()
    try:
        for p in [_VECTOR_ROOT, _COLLINS_SRC]:
            if p not in sys.path:
                sys.path.insert(0, p)

        os.chdir(_VECTOR_ROOT)

        from retrieval.vector_adapter import retrieve as collins_retrieve
        result = collins_retrieve(query=query, top_k=top_k)

        chunks = []
        for r in result.get("results", []):
            source = r.get("source", "")
            if source.endswith("_ocr.txt"):
                source = source.replace("_ocr.txt", ".pdf")
            source = re.sub(r'^[0-9a-f]{8}_', '', source)

            chunks.append({
                "chunk_id": r.get("chunk_id", ""),
                "document_id": r.get("document_id", ""),
                "source_name": source,
                "text": r.get("text", ""),
                "score": r.get("score", 0.0),
                "rank":  r.get("rank", 0),
                "document_title": r.get("document_title", "").replace("_ocr", "").replace("Ocr", "").strip(),
                "source": source,
                "citation": r.get("citation", ""),
                "metadata": r.get("metadata", {}),
            })
        return chunks

    except FileNotFoundError:
        raise FileNotFoundError("No FAISS index found — upload documents first")
    except Exception as e:
        raise RuntimeError(f"Collins vector retrieval failed: {e}")
    finally:
        for p in [_VECTOR_ROOT, _COLLINS_SRC]:
            if p in sys.path:
                sys.path.remove(p)
        os.chdir(original_dir)