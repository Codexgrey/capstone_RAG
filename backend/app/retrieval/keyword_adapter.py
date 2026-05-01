"""
app/retrieval/keyword_adapter.py  —  Olivier's BM25 keyword retrieval
"""
import os, re
from app.retrieval.module_loader import load_adapter

# RAG_PROJECT_ROOT must point to the repo root (where backend/, keyword_retrieval/ etc. live)
# Set it in backend/.env — see .env.example
_PROJECT = os.environ.get(
    "RAG_PROJECT_ROOT",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../"))
)
_ROOT = os.path.join(_PROJECT, "keyword_retrieval")
_ADAPTER = os.path.join(_ROOT, "src", "retrieval", "keyword_adapter.py")


def _get():
    return load_adapter(_ADAPTER, _ROOT)


def ingest(file_paths, chunk_size=300, chunk_overlap=50):
    original = os.getcwd()
    try:
        os.chdir(_ROOT)
        return _get().ingest(file_paths=file_paths, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    except Exception as e:
        return {"status": "error", "documents_ingested": 0, "total_chunks": 0, "latency_ms": 0.0, "error": str(e)}
    finally:
        os.chdir(original)


def retrieve(query, top_k=5):
    original = os.getcwd()
    try:
        os.chdir(_ROOT)
        result = _get().retrieve(query=query, top_k=top_k)
        return _normalise(result.get("results", []))
    except FileNotFoundError:
        raise FileNotFoundError("No keyword index found — upload documents first")
    except Exception as e:
        raise RuntimeError(f"Olivier keyword retrieval failed: {e}")
    finally:
        os.chdir(original)


def _normalise(results):
    out = []
    for i, r in enumerate(results):
        src = re.sub(r'^[0-9a-f]{8}_', '', (r.get("source") or r.get("source_name") or "Unknown")
                     .replace("_ocr.txt", ".pdf").replace("_ocr", ""))
        out.append({**r, "source_name": src, "source": src,
                    "score": float(r.get("bm25_score") or r.get("score") or 0.0),
                    "rank": r.get("rank") or (i + 1)})
    return out