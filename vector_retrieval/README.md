# vector_retrieval

Vector-based retrieval module for the Capstone RAG System.  
Uses `sentence-transformers` (`all-MiniLM-L6-v2`) for dense embeddings and `FAISS` for similarity search.

**Branch:** `feat/vector-retrieval-collins`  
**Contract:** `shared_data/schemas/retrieval_response.schema.json`

---

## Structure

```
vector_retrieval/
│
├── src/
│   ├── models/
│   │   └── embedding_model.py       # loads & encodes with sentence-transformers
│   │
│   ├── indexing/
│   │   ├── indexer.py               # orchestrates load → chunk → embed → index → persist
│   │   └── vector_store.py          # FAISS index build, save, and reload
│   │
│   ├── retrieval/
│   │   ├── retriever.py             # core similarity search, returns top-k chunks
│   │   └── vector_adapter.py        # backend-facing adapter — ingest() + retrieve()
│   │
│   ├── evaluation/
│   │   └── evaluate.py              # runtime report + Precision@K, Recall@K, MRR
│   │
│   ├── utils/
│   │   ├── loader.py                # multi-format document loader (.txt, .pdf, .docx, .md)
│   │   ├── chunker.py               # overlapping word-level chunker with metadata
│   │   ├── prompts.py               # builds structured RAG prompt
│   │   └── response_printer.py      # formats and prints the final retrieval report
│   │
│   └── main.py                      # local entry point (research/testing only)
│
├── tests/                           # sample documents for local testing
│
├── .env                             # API keys (not committed)
├── requirements.txt
└── README.md
```

---

## Pipeline

```
[Backend: user uploads files]
    ↓
ingest(file_paths)
    ↓
Loader      — auto-detects .txt / .pdf / .docx / .md
    ↓
Chunker     — overlapping word-level chunks with file metadata
    ↓
Embedder    — all-MiniLM-L6-v2 via sentence-transformers
    ↓
FAISS Index — built and saved to faiss_index.bin + chunk_records.npy
    ↓
[Backend: user submits query]
    ↓
retrieve(query, top_k)
    ↓
Similarity search → top-k chunks → contract-compliant response dict

[Local research only: Prompt Builder → Groq Generator → Report → Evaluation]
```

---

## Setup

**1. Install dependencies**
```bash
pip install -r requirements.txt
```

**2. Configure your API key**

Create a `.env` file in the `vector_retrieval/` root:
```
GROQ_API_KEY=your_groq_api_key_here
```

**3. Add documents (local testing only)**

Place any `.txt`, `.pdf`, `.docx`, or `.md` files into the `tests/` folder.  
The pipeline scans the folder automatically when running `src/main.py`.  
In the integrated system, the backend provides file paths directly via `ingest()`.

---

## Run (local research mode)

From the `vector_retrieval/` root:
```bash
python -m src.main
```

To change the test query, edit `QUERY` in `src/main.py`.  
To change chunk size or overlap, edit `CHUNK_SIZE` and `CHUNK_OVERLAP` in `src/main.py`.  
To reload a saved index without re-embedding, set `REBUILD_INDEX = False` in `src/main.py`.

---

## Backend Integration

The adapter in `src/retrieval/vector_adapter.py` is the **only file the backend should import**.  
It exposes two functions that together form the complete plug-in interface:

### 1. `ingest()` — called after a user uploads documents

```python
from vector_retrieval.src.retrieval.vector_adapter import ingest

result = ingest(
    file_paths=["/tmp/uploads/report.pdf", "/tmp/uploads/notes.txt"],
    chunk_size=300,       # optional, default: 300
    chunk_overlap=50,     # optional, default: 50
)
```

**Returns:**
```python
{
    "status":             "ok",          # "ok" | "error"
    "documents_ingested": 2,
    "total_chunks":       94,
    "index_path":         "faiss_index.bin",
    "chunks_path":        "chunk_records.npy",
    "latency_ms":         1823.4,
    "error":              None            # str on failure, None on success
}
```

### 2. `retrieve()` — called when a user submits a query

```python
from vector_retrieval.src.retrieval.vector_adapter import retrieve

response = retrieve(query="What is RAG?", top_k=3)
```

**Returns** (matches `shared_data/schemas/retrieval_response.schema.json`):
```python
{
    "query":      "What is RAG?",
    "method":     "vector",
    "results": [
        {
            "rank":           1,
            "chunk_id":       "doc-001-chunk-4",
            "document_id":    "doc-001",
            "document_title": "Feasibility Study Report",
            "source":         "feasibility_study_report.pdf",
            "text":           "...",
            "score":          0.91,
            "citation":       "[Feasibility Study Report | doc-001-chunk-4]",
            "metadata": {
                "file_name":    "feasibility_study_report.pdf",
                "file_type":    "pdf",
                "file_size_kb": 245.6,
                "uploaded_at":  "2026-04-20T10:30:00+00:00"
            }
        }
    ],
    "latency_ms": 42.9
}
```

### How the adapter manages state

The adapter loads the embedding model and FAISS index into memory on the first `retrieve()` call and reuses them for all subsequent calls. After `ingest()` rebuilds the index, the adapter automatically reloads it on the next `retrieve()` call. The backend does not need to manage any of this — it just calls `ingest()` then `retrieve()`.

### Environment variables for backend deployment

| Variable | Default | Description |
|---|---|---|
| `VECTOR_INDEX_PATH` | `faiss_index.bin` | Path to saved FAISS index |
| `VECTOR_CHUNKS_PATH` | `chunk_records.npy` | Path to saved chunk records |
| `VECTOR_MODEL_NAME` | `all-MiniLM-L6-v2` | Embedding model identifier |

---

## Evaluation

The evaluation module runs automatically after every `python -m src.main` run.  
It produces two reports:

- **Runtime report** — always fires. Shows per-rank similarity scores, source coverage, and latency.
- **Ground-truth report** — fires when the query is registered in `TEST_QUERIES` inside `evaluate.py`. Reports Precision@K, Recall@K, and MRR.

To add a query to the ground-truth bank, edit `TEST_QUERIES` in `src/evaluation/evaluate.py`.

---

## Notes

- The Groq generator in `src/main.py` is **research scaffolding only**. In the integrated system, all LLM calls belong in `backend/app/generation/llm_client.py`.
- The `tests/` folder is for local development only. It has no effect on backend operation — the backend always provides file paths explicitly via `ingest()`.
- Supported file types: `.txt`, `.pdf`, `.docx`, `.md`