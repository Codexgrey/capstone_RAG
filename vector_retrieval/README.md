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
│   │   └── vector_adapter.py        # backend-facing adapter — retrieve(query, top_k)
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
│   ├── sample.txt
│   └── sample.md
│
├── .env                             # API keys (not committed)
├── requirements.txt
└── README.md
```

---

## Pipeline

```
Documents (tests/ folder)
    ↓
Loader      — auto-detects .txt / .pdf / .docx / .md
    ↓
Chunker     — overlapping word-level chunks with file metadata
    ↓
Embedder    — all-MiniLM-L6-v2 via sentence-transformers
    ↓
FAISS Index — built and saved to faiss_index.bin + chunk_records.npy
    ↓
Retriever   — similarity search → top-k chunks
    ↓
[Research only: Prompt Builder → Groq Generator → Report → Evaluation]
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

**3. Add documents**

Place any `.txt`, `.pdf`, `.docx`, or `.md` files into the `tests/` folder.  
The pipeline scans the folder automatically — no manual path configuration needed.

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

For backend integration, import the adapter — not `main.py`:

```python
from vector_retrieval.src.retrieval.vector_adapter import retrieve

response = retrieve(query="What is RAG?", top_k=3)
```

The adapter handles model loading and index loading internally.  
It returns a dict matching `shared_data/schemas/retrieval_response.schema.json`:

```python
{
    "query":      "What is RAG?",
    "method":     "vector",
    "results":    [
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

**Environment variables for backend deployment:**

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
- The FAISS index and chunk records are saved to disk after every ingestion run. Set `REBUILD_INDEX = False` in `main.py` to reload them and skip re-embedding.
- Supported file types: `.txt`, `.pdf`, `.docx`, `.md`
