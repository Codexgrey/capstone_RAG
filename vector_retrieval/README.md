# Vector Retrieval — FAISS Semantic Search

Collins's vector retrieval module. Uses `all-MiniLM-L6-v2` sentence embeddings
and FAISS for fast similarity search.

## Setup

```bash
cd vector_retrieval
pip install -r requirements.txt
```

## Run Standalone (Research / Testing)

```bash
cd vector_retrieval
python src/main.py
```

`main.py` provides an interactive pipeline: load document → chunk → embed → index → query.

## Backend Integration

The backend calls two functions from `src/retrieval/vector_adapter.py`:

```python
from retrieval.vector_adapter import ingest, retrieve

# After a user uploads a file:
ingest(file_paths=["path/to/doc.pdf"], chunk_size=400, chunk_overlap=50)

# On a query:
result = retrieve(query="What is RAG?", top_k=5)
```

`ingest()` saves `faiss_index.bin` and `chunk_records.npy` to the
`vector_retrieval/` directory. `retrieve()` loads them on first call.

## Pipeline

```
Document → Loader → Chunker (overlap) → all-MiniLM-L6-v2 → FAISS → Retriever
```

## Structure

```
src/
├── models/         embedding_model.py    — SentenceTransformer wrapper
├── indexing/       indexer.py            — build + save FAISS index
│                   vector_store.py       — load FAISS index
├── retrieval/      retriever.py          — similarity search
│                   vector_adapter.py     — backend plug-in interface
├── evaluation/     evaluate.py           — precision@k, recall@k, MRR
└── utils/          chunker.py, loader.py, prompts.py, response_printer.py
```

## Evaluation

```bash
cd vector_retrieval
python src/evaluation/evaluate.py
```

Metrics: `precision_at_k`, `recall_at_k`, `mean_reciprocal_rank`.

## Environment Variables (optional overrides)

| Variable             | Default              |
|----------------------|----------------------|
| `VECTOR_INDEX_PATH`  | `faiss_index.bin`    |
| `VECTOR_CHUNKS_PATH` | `chunk_records.npy`  |
| `VECTOR_MODEL_NAME`  | `all-MiniLM-L6-v2`   |
