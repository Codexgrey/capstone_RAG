# Retrieval Module API Contract
**Version:** 1.0 — derived from `vector_retrieval` (iteration G)  
**Authority:** `3. Highlevel_Project_Overview.pdf`  
**Applies to:** `vector_retrieval/`, `keyword_retrieval/`, `clara_retrieval/`

---

## Purpose

This document defines the shared interface contract that all three retrieval modules must satisfy for integration with the backend adapter layer (`backend/app/retrieval/`). The internals of each module differ — FAISS similarity search, BM25 scoring, CLaRa-style reranking — but the input they accept and the output they return must be identical in shape.

---

## The Rule

> **Every retrieval module exposes one public function:**
> ```python
> def retrieve(query: str, top_k: int = 3) -> dict
> ```
> It accepts a query string and returns a response dict matching `retrieval_response.schema.json`.

The backend adapter calls this function. It does not know or care about FAISS, BM25, or ColBERT internals.

---

## 1. Retrieval Request

The backend passes these parameters when calling a retrieval module:

```python
{
    "query":   str,        # natural language question — required
    "top_k":   int = 3,    # number of chunks to return — default 3
    "method":  str,        # "vector" | "keyword" | "clara"
    "filters": {           # optional
        "document_ids": list[str],   # restrict to these doc IDs
        "file_types":   list[str]    # restrict to these file types
    },
    "options": {           # optional
        "use_reranking": bool        # CLaRa reranking pass (default False)
    }
}
```

Full schema: `shared_data/schemas/retrieval_request.schema.json`

---

## 2. Retrieval Response

Every module returns a dict in this exact shape:

```python
{
    "query":      str,          # original query echoed back
    "method":     str,          # "vector" | "keyword" | "clara"
    "results":    list[dict],   # top-k chunks — see below
    "latency_ms": float         # retrieval time in milliseconds
}
```

Each item in `results`:

```python
{
    "rank":           int,    # 1-based (1 = best match)
    "chunk_id":       str,    # e.g. "doc-001-chunk-4"
    "document_id":    str,    # e.g. "doc-001"
    "document_title": str,    # e.g. "Feasibility Study Report"
    "source":         str,    # original filename
    "text":           str,    # chunk text content
    "score":          float,  # relevance score — see note below
    "citation":       str,    # "[Document Title | chunk_id]"
    "metadata": {
        "file_name":    str,
        "file_type":    str,   # "txt" | "pdf" | "docx" | "md"
        "file_size_kb": float,
        "uploaded_at":  str    # ISO 8601 UTC
    }
}
```

> **Score note:** Score meaning differs per method but direction is consistent — higher always means more relevant.
> - Vector → L2-derived similarity in (0, 1]
> - Keyword → BM25 score (unbounded positive)
> - CLaRa → method-specific confidence

Full schema: `shared_data/schemas/retrieval_response.schema.json`

---

## 3. Chunk Schema

All modules index documents using the same chunk record structure. The backend ingestion layer (`backend/app/ingestion/`) produces chunks in this format and passes them to each retrieval module's indexer.

Key required fields:

| Field | Type | Description |
|---|---|---|
| `chunk_id` | string | `"<doc_id>-chunk-<N>"` |
| `document_id` | string | `"doc-<NNN>"` (zero-padded) |
| `document_title` | string | Title-cased from filename |
| `source` | string | Original filename |
| `text` | string | Chunk text content |
| `metadata.file_type` | string | `txt`, `pdf`, `docx`, `md` |
| `metadata.uploaded_at` | string | ISO 8601 UTC timestamp |

Full schema: `shared_data/schemas/chunk.schema.json`

---

## 4. LLM Generation Input

The backend generation layer (`backend/app/generation/llm_client.py`) receives the retrieval response and builds the LLM prompt. The prompt is constructed from:

```python
{
    "query":             str,         # original user query
    "retrieval_method":  str,         # "vector" | "keyword" | "clara"
    "retrieved_chunks":  list[dict]   # results list from retrieval response
}
```

Each chunk passed to the LLM includes: `chunk_id`, `text`, `document_title`, `source`, `citation`.

> **Architectural rule:** All LLM interaction belongs in `backend/app/generation/llm_client.py`. Retrieval modules do not call any LLM. The generator in `vector_retrieval/src/main.py` is research scaffolding only.

---

## 5. Final Answer Output

The backend returns this to the frontend (`backend/app/api/query.py` response):

```python
{
    "query":            str,
    "answer":           str,         # LLM-generated answer (2-4 sentences)
    "evidence_used":    list[dict],  # chunk_id + contribution per chunk
    "citations":        list[dict],  # chunk_id, document_title, source, file_type
    "retrieval_method": str,
    "latency_ms":       float        # total latency (retrieval + generation)
}
```

Full schema: `shared_data/schemas/answer_response.schema.json`

---

## 6. Module Responsibilities Summary

| Module | Owns | Does NOT own |
|---|---|---|
| `vector_retrieval/` | FAISS index, MiniLM embeddings, similarity search | LLM calls, prompt building (in production) |
| `keyword_retrieval/` | BM25 index, tokenisation, keyword scoring | LLM calls, prompt building |
| `clara_retrieval/` | ColBERT-style encoding, reranking logic | LLM calls, prompt building |
| `backend/app/retrieval/` | Adapter layer — calls each module's `retrieve()` | Retrieval internals |
| `backend/app/generation/` | Prompt building, LLM client, response formatting | Retrieval |
| `backend/app/ingestion/` | Document parsing, chunking, metadata extraction | Retrieval, generation |

---

## 7. Integration Checklist

Before submitting your retrieval module for backend integration, verify:

- [ ] `retrieve(query, top_k)` function is importable from your module
- [ ] Response matches `retrieval_response.schema.json` exactly
- [ ] `score` field is present and higher = more relevant
- [ ] `citation` field uses format `[Document Title | chunk_id]`
- [ ] `metadata.file_type` and `metadata.uploaded_at` present on every result
- [ ] `latency_ms` is measured and included in the response
- [ ] Module does not make any LLM API calls
