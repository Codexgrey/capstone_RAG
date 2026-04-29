"""
app/retrieval/vector_adapter.py
================================
Bridge between Khalid's backend and Collins's vector_retrieval module.

Assumes repo structure:
    capstone_RAG/
      backend/              ← you are here
      vector_retrieval/
        src/                ← Collins's code lives here
"""

import sys
import os

# Add both vector_retrieval/ root AND vector_retrieval/src/ to path
_VECTOR_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../../vector_retrieval")
)
_COLLINS_SRC = os.path.join(_VECTOR_ROOT, "src")

for p in [_VECTOR_ROOT, _COLLINS_SRC]:
    if p not in sys.path:
        sys.path.insert(0, p)


def ingest(file_paths: list, chunk_size: int = 300, chunk_overlap: int = 50) -> dict:
    try:
        # Set working directory to vector_retrieval/ so Collins's
        # relative paths (faiss_index.bin, chunk_records.npy) resolve correctly
        original_dir = os.getcwd()
        os.chdir(_VECTOR_ROOT)

        from retrieval.vector_adapter import ingest as collins_ingest
        result = collins_ingest(
            file_paths=file_paths,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        return result
    except Exception as e:
        return {"status": "error", "error": str(e),
                "documents_ingested": 0, "total_chunks": 0}
    finally:
        os.chdir(original_dir)   # always restore original directory


def retrieve(query: str, top_k: int = 5) -> list:
    try:
        original_dir = os.getcwd()
        os.chdir(_VECTOR_ROOT)

        from retrieval.vector_adapter import retrieve as collins_retrieve
        result = collins_retrieve(query=query, top_k=top_k)

        chunks = []
        for r in result.get("results", []):
            # Clean up OCR txt filename — show original PDF name
            source = r.get("source", "")
            if source.endswith("_ocr.txt"):
                source = source.replace("_ocr.txt", ".pdf")
            # Also strip the uuid prefix (e.g. "59a1edc2_filename.pdf" → "filename.pdf")
            import re
            source_clean = re.sub(r'^[0-9a-f]{8}_', '', source)

            chunks.append({
                "chunk_id":       r.get("chunk_id", ""),
                "document_id":    r.get("document_id", ""),
                "source_name":    source_clean,      # ← clean name
                "text":           r.get("text", ""),
                "score":          r.get("score", 0.0),
                "rank":           r.get("rank", 0),
                "document_title": r.get("document_title", "").replace("_ocr", "").replace("Ocr", "").strip(),
                "source":         source_clean,      # ← clean name
                "citation":       r.get("citation", ""),
                "metadata":       r.get("metadata", {}),
            })
        return chunks

    except FileNotFoundError:
        raise FileNotFoundError("No FAISS index found — upload documents first")
    except Exception as e:
        raise RuntimeError(f"Collins vector retrieval failed: {e}")
    finally:
        os.chdir(original_dir)