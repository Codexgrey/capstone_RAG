# adapter_contract.md

**Scope:** All three retrieval modules — `vector_retrieval`, `keyword_retrieval`, `clara_retrieval`  
**Author:** Collins Ovuakporaye — feat/vector-retrieval-collins  
**For:** Olivier, Nathan (Retrieval Researchers) — implement this interface in your module  
**For:** Khalid (Backend) — call this interface from `backend/app/retrieval/`  
**Related schemas:** `shared_data/schemas/retrieval_request.schema.json`, `shared_data/schemas/retrieval_response.schema.json`

---

## Overview

Every retrieval module must expose an adapter file at:

```
<module_name>/src/retrieval/<module_name>_adapter.py
```

For example:
- `vector_retrieval/src/retrieval/vector_adapter.py` ✓ implemented
- `keyword_retrieval/src/retrieval/keyword_adapter.py`
- `clara_retrieval/src/retrieval/clara_adapter.py`

The adapter is the **only file the backend imports from any retrieval module**. It exposes two functions that together form the complete plug-in interface:

```python
ingest(file_paths, chunk_size, chunk_overlap) -> dict
retrieve(query, top_k) -> dict
```

The backend calls `ingest()` when a user uploads documents and `retrieve()` when a user submits a query. Internals differ between modules — FAISS, BM25, ColBERT, whatever each approach requires — but the two function signatures and their return shapes are non-negotiable.

---

## 1. `ingest(file_paths, chunk_size, chunk_overlap)`

Called by the backend after receiving uploaded files. Runs the full ingestion pipeline for the module's chosen retrieval approach and persists the index to disk. After this call, `retrieve()` will search the newly built index.

### Signature

```python
def ingest(
    file_paths: list,
    chunk_size: int = 300,
    chunk_overlap: int = 50,
) -> dict
```

### Parameters

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `file_paths` | list[str] | Yes | — | Absolute or relative paths to uploaded files. Supported: `.txt`, `.md`, `.pdf`, `.docx` |
| `chunk_size` | int | No | 300 | Number of words per chunk |
| `chunk_overlap` | int | No | 50 | Number of overlapping words between adjacent chunks |

### Returns

```python
{
    "status":             "ok",     # "ok" | "error"
    "documents_ingested": 2,        # number of files successfully processed
    "total_chunks":       94,       # total chunks written to the index
    "index_path":         "...",    # path to the persisted index file
    "chunks_path":        "...",    # path to the persisted chunk records
    "latency_ms":         1823.4,   # full ingestion time in milliseconds
    "error":              None      # str describing failure, None on success
}
```

`index_path` and `chunks_path` will differ between modules depending on the underlying index format used. The shape of the response dict must be identical across all three.

### Reference implementation (vector)

```python
from vector_retrieval.src.retrieval.vector_adapter import ingest

result = ingest(file_paths=["/tmp/uploads/report.pdf", "/tmp/uploads/notes.txt"])

if result["status"] == "ok":
    # index is ready — queries can now be served
    pass
else:
    # handle result["error"]
    pass
```

---

## 2. `retrieve(query, top_k)`

Called by the backend when the user submits a query. Loads the model and index on first call and reuses them for all subsequent calls. Always searches the most recently ingested index.

Return shape must match `shared_data/schemas/retrieval_response.schema.json` exactly.

### Signature

```python
def retrieve(
    query: str,
    top_k: int = 3,
) -> dict
```

### Parameters

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `query` | str | Yes | — | Natural language question from the user |
| `top_k` | int | No | 3 | Number of top chunks to return |

### Returns

```python
{
    "query":   "What is RAG?",
    "method":  "vector",        # "vector" | "keyword" | "clara" — set by each module
    "results": [
        {
            "rank":           1,
            "chunk_id":       "doc-001-chunk-4",
            "document_id":    "doc-001",
            "document_title": "Feasibility Study Report",
            "source":         "feasibility_study_report.pdf",
            "text":           "...",
            "score":          0.91,    # see score note below
            "citation":       "[Feasibility Study Report | doc-001-chunk-4]",
            "metadata": {
                "file_name":    "feasibility_study_report.pdf",
                "file_type":    "pdf",
                "file_size_kb": 245.6,
                "uploaded_at":  "2026-04-20T10:30:00+00:00"
            }
        }
        # ... up to top_k results
    ],
    "latency_ms": 42.9
}
```

**Score note:** The `score` field must be present and higher must mean more relevant. The internal computation differs per module — L2-derived similarity for vector, BM25 score for keyword, method-specific confidence for CLaRa. The contract requires the field, not a specific scale.

### Reference implementation (vector)

```python
from vector_retrieval.src.retrieval.vector_adapter import retrieve

response = retrieve(query="What are the risks of RAG hallucination?", top_k=3)
# pass response["results"] to backend/app/generation/llm_client.py
```

---

## 3. State management

Each adapter manages its own internal state. The backend does not pass model objects, index handles, or chunk records.

| Behaviour | Requirement |
|---|---|
| Model loading | Load once on first `retrieve()` call, reuse across all subsequent calls |
| Index loading | Load on first `retrieve()` call after startup, or after any `ingest()` call |
| Re-ingestion | After `ingest()` rebuilds the index, the next `retrieve()` call must search the new index |

---

## 4. Environment variables

Each module should make its index paths and model name configurable via environment variables. Naming convention:

| Variable pattern | Example (vector) | Description |
|---|---|---|
| `<MODULE>_INDEX_PATH` | `VECTOR_INDEX_PATH` | Path to the persisted index |
| `<MODULE>_CHUNKS_PATH` | `VECTOR_CHUNKS_PATH` | Path to the persisted chunk records |
| `<MODULE>_MODEL_NAME` | `VECTOR_MODEL_NAME` | Model identifier |

These must be set consistently in the backend's deployment environment.

---

## 5. Error handling

| Condition | Required behaviour |
|---|---|
| `ingest()` — pipeline failure | Return `{"status": "error", "error": "<message>"}`, do not raise |
| `retrieve()` — empty query | Raise `ValueError` |
| `retrieve()` — index not found | Raise `FileNotFoundError` |

The backend must guard `retrieve()` calls until at least one successful `ingest()` has been completed, and return an appropriate HTTP 400/503 to the frontend if called before that.

---

## 6. Integration checklist

For each retrieval module, verify all of the following before handing off to the backend:

| | Item |
|---|---|
| ☐ | `ingest(file_paths)` and `retrieve(query, top_k)` are both importable from your adapter |
| ☐ | `ingest()` called and returns `status: "ok"` before any `retrieve()` call is made |
| ☐ | `retrieve()` response matches `retrieval_response.schema.json` exactly |
| ☐ | `score` field is present and higher = more relevant |
| ☐ | `citation` field uses format `[Document Title | chunk_id]` |
| ☐ | `metadata.file_type` and `metadata.uploaded_at` present on every result |
| ☐ | `latency_ms` measured and included in the response |
| ☐ | Module makes no LLM API calls |
| ☐ | No other file in your module is imported by the backend |
| ☐ | README.md documents how to run your module locally |
