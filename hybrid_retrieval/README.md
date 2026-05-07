# Hybrid Retrieval — FAISS + BM25 + RRF

Nathan's hybrid retrieval module. Combines vector and keyword retrieval
using Reciprocal Rank Fusion (RRF) for a single unified ranking.

## Setup

```bash
cd hybrid_retrieval
pip install -r requirements.txt
```

Note: requires `tesseract-ocr` for OCR fallback on scanned PDFs:
- Ubuntu: `sudo apt install tesseract-ocr`
- macOS: `brew install tesseract`
- Windows: https://github.com/UB-Mannheim/tesseract/wiki

## Run Standalone (Research / Testing)

```bash
cd hybrid_retrieval
python src/main.py
```

Place test documents in `src/content/` — the script auto-discovers them.

## Backend Integration

The backend calls two functions from `src/retrieval/hybrid_adapter.py`:

```python
# Loaded by backend's module_loader dynamically
ingest(file_paths=["path/to/doc.pdf"], chunk_size=150, chunk_overlap=30)
result = retrieve(query="What is RAG?", top_k=5, rrf_k=60)
```

Persists to: `hybrid_faiss_index.bin` and `hybrid_chunk_records.npy`
(in the `hybrid_retrieval/` directory when called from the backend).

## Pipeline

```
Document → Loader (OCR fallback) → Chunker →
  ┌─ FAISS embedding (all-MiniLM-L6-v2) ─┐
  └─ BM25 tokenisation + scoring ─────────┘
         ↓
  Reciprocal Rank Fusion (RRF)
         ↓
  Merged top-k results
```

## RRF Formula

```
score = 1/(k + rank_vector) + 1/(k + rank_bm25)
```
Default `k=60` (standard literature value). Each result shows whether it was
found by vector only, keyword only, or both.

## Structure

```
src/
├── models/         embedding_model.py    — SentenceTransformer wrapper
├── preprocessing/  preprocess.py         — language detection, tokenisation
├── indexing/       vector_store.py       — FAISS build/save/load
│                   bm25_indexer.py       — BM25 + inverted index
├── retrieval/      vector_retriever.py   — FAISS search
│                   bm25_retriever.py     — BM25 search
│                   hybrid_retriever.py   — RRF fusion
│                   hybrid_adapter.py     — backend plug-in interface
└── utils/          chunker.py, loader.py, prompts.py
```

## Environment Variables (optional overrides)

| Variable                   | Default                    |
|----------------------------|----------------------------|
| `HYBRID_VECTOR_INDEX_PATH` | `hybrid_faiss_index.bin`   |
| `HYBRID_VECTOR_CHUNKS_PATH`| `hybrid_chunk_records.npy` |
| `HYBRID_MODEL_NAME`        | `all-MiniLM-L6-v2`         |
