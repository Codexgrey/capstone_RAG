# Retrieval Module API Contract

**Version:** 2.0 — updated to reflect `hybrid_retrieval` replacing `clara_retrieval`  
**Authority:** `3. Highlevel_Project_Overview.pdf`  
**Applies to:** `vector_retrieval/`, `keyword_retrieval/`, `hybrid_retrieval/`

---

## Purpose

This document defines the shared interface that all three retrieval modules must satisfy for integration with the backend adapter layer (`backend/app/retrieval/`). The internals differ — FAISS similarity search, BM25 scoring, FAISS+BM25+RRF fusion — but the input accepted and output returned must be identical in shape.

---

## The Rule

> **Every retrieval module exposes one adapter file with two public functions:**
> ```python
> ingest(file_paths, chunk_size, chunk_overlap) -> dict
> retrieve(query, top_k) -> dict
> ```
> The backend adapter calls these functions. It does not know or care about FAISS, BM25, or RRF internals.

---

## 1. Retrieval Request

The backend passes these parameters when calling a retrieval module:

```python
{
    "query":   str,        # natural language question — required
    "top_k":   int = 3,    # number of chunks to return — default 3
    "method":  str,        # "vector" | "keyword" | "hybrid"
    "filters": {           # optional
        "document_ids": list[str],
        "file_types":   list[str]
    },
    "options": {           # optional
        "use_reranking": bool   # hybrid reranking pass (default False)
    }
}
```

---

## 2. Retrieval Response

Every retrieval module returns this shape (matching `retrieval_response.schema.json`):

```python
{
    "query":      str,         # original query echoed back
    "method":     str,         # "vector" | "keyword" | "hybrid"
    "results":    list[dict],  # top-k chunks, rank 1 = best
    "latency_ms": float
}
```

Each result:

```python
{
    "rank":           int,
    "chunk_id":       str,    # "doc-001-chunk-4"
    "document_id":    str,    # "doc-001"
    "document_title": str,
    "source":         str,    # original filename
    "text":           str,
    "score":          float,  # higher = more relevant
    "citation":       str,    # "[Doc Title | chunk_id]"
    "metadata":       dict    # file_name, file_type, uploaded_at
}
```

---

## 3. Ingest Request / Response

```python
# Call
ingest(file_paths=["path/to/doc.pdf"], chunk_size=300, chunk_overlap=50)

# Response
{
    "status":             "ok" | "error",
    "documents_ingested": int,
    "total_chunks":       int,
    "latency_ms":         float,
    "error":              str | None
}
```

---

## 4. Final Answer Output

After retrieval, the backend builds a prompt and calls the LLM. The final answer matches `answer_response.schema.json`:

```python
{
    "query":             str,
    "answer":            str,
    "evidence_used":     list[dict],   # chunk_id + contribution preview
    "citations":         list[dict],   # chunk_id, document_title, source, file_type
    "retrieval_method":  str,          # "vector" | "keyword" | "hybrid"
    "latency_ms":        float,
    "session_id":        str
}
```

---

## 5. Module Responsibilities

| Module             | Owns                                                      | Does NOT own                    |
|--------------------|-----------------------------------------------------------|---------------------------------|
| `vector_retrieval/`| FAISS index, sentence-transformer embeddings, similarity  | LLM calls, prompt building      |
| `keyword_retrieval/`| BM25 model, inverted index, tokenisation, stemming       | LLM calls, prompt building      |
| `hybrid_retrieval/`| FAISS + BM25 + RRF fusion, combined ranking               | LLM calls, prompt building      |
| `backend/`         | Document ingestion, routing, LLM generation, persistence  | Retrieval algorithms            |

---

## 6. Shared Chunk Schema

All three modules index documents using the same chunk structure (`chunk.schema.json`):

```python
{
    "chunk_id":       str,   # "doc-001-chunk-4"
    "document_id":    str,   # "doc-001"
    "document_title": str,
    "source":         str,   # original filename
    "text":           str,
    "chunk_index":    int,
    "word_count":     int,
    "metadata": {
        "file_name":    str,
        "file_type":    str,   # "pdf" | "txt" | "docx" | "md"
        "file_size_kb": float,
        "uploaded_at":  str    # ISO 8601
    }
}
```
