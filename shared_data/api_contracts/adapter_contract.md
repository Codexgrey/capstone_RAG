# Adapter Contract

**Scope:** All three retrieval modules — `vector_retrieval`, `keyword_retrieval`, `hybrid_retrieval`  
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

Implemented adapters:
- `vector_retrieval/src/retrieval/vector_adapter.py`   ✓
- `keyword_retrieval/src/retrieval/keyword_adapter.py` ✓
- `hybrid_retrieval/src/retrieval/hybrid_adapter.py`   ✓

The adapter is the **only file the backend imports from any retrieval module**. It exposes two functions forming the complete plug-in interface:

```python
ingest(file_paths, chunk_size, chunk_overlap) -> dict
retrieve(query, top_k) -> dict
```

The backend calls `ingest()` when a user uploads documents and `retrieve()` when a user submits a query. Internals differ — FAISS, BM25, FAISS+BM25+RRF — but the two function signatures and return shapes are non-negotiable.

---

## 1. `ingest(file_paths, chunk_size, chunk_overlap)`

Called by the backend after receiving uploaded files. Runs the full ingestion pipeline and persists the index to disk. After this call, `retrieve()` will search the newly built index.

### Signature

```python
def ingest(
    file_paths:    list,   # absolute or relative file paths — .txt, .md, .pdf, .docx
    chunk_size:    int = 300,
    chunk_overlap: int = 50,
) -> dict:
```

### Return shape

```python
{
    "status":             "ok" | "error",
    "documents_ingested": int,
    "total_chunks":       int,
    "latency_ms":         float,
    "error":              str | None,   # set when status == "error"
}
```

---

## 2. `retrieve(query, top_k)`

Called by the backend for every user query. Searches the most recently ingested index and returns the top-k matching chunks.

### Signature

```python
def retrieve(
    query: str,
    top_k: int = 3,
) -> dict:
```

### Return shape — matches `retrieval_response.schema.json`

```python
{
    "query":      str,         # original query echoed back
    "method":     str,         # "vector" | "keyword" | "hybrid"  — set by each module
    "results":    list[dict],  # top-k chunks, ordered by score descending
    "latency_ms": float,
}
```

Each result dict:

```python
{
    "rank":           int,    # 1-based (rank 1 = best match)
    "chunk_id":       str,    # e.g. "doc-001-chunk-4"
    "document_id":    str,    # e.g. "doc-001"
    "document_title": str,    # human-readable title
    "source":         str,    # original filename
    "text":           str,    # chunk content
    "score":          float,  # relevance score — higher is better
    "citation":       str,    # "[Doc Title | chunk_id]"
    "metadata":       dict,   # file_name, file_type, uploaded_at
}
```

---

## 3. Score semantics

| Module  | Score field   | Meaning                          |
|---------|---------------|----------------------------------|
| vector  | `score`       | L2-derived similarity in (0, 1]  |
| keyword | `bm25_score`  | BM25 score (unbounded, ≥ 0)      |
| hybrid  | `rrf_score`   | RRF-fused score (0 to ~0.033)    |

The backend adapter normalises all scores so `score` in the shared response is always higher = more relevant.

---

## 4. Error handling

`ingest()` never raises — it returns `{"status": "error", "error": "..."}`.  
`retrieve()` raises `FileNotFoundError` if the index does not exist (ingest not yet called).  
The backend catches both cases and falls back to PostgreSQL chunks.
