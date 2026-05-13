# Keyword Retrieval — BM25 Search

Olivier's keyword retrieval module. Uses BM25Okapi for lexical search
with multi-language support via NLTK.

## Setup

```bash
cd keyword_retrieval
pip install -r requirements.txt
```

Download NLTK data on first run (automatic) or manually:
```python
import nltk
nltk.download('stopwords')
nltk.download('punkt')
```

## Run Standalone (Research / Testing)

```bash
cd keyword_retrieval/src
python main.py
```

Interactive menu drives the full 10-step pipeline.

## Backend Integration

The backend calls two functions from `src/retrieval/keyword_adapter.py`:

```python
# Loaded by backend's module_loader dynamically
ingest(file_paths=["path/to/doc.pdf"], chunk_size=300, chunk_overlap=50)
result = retrieve(query="What is RAG?", top_k=5)
```

Persists index to: `keyword_bm25.pkl`, `keyword_index.pkl`, `keyword_chunks.pkl`
(in the `keyword_retrieval/` directory when called from the backend).

## Pipeline

```
Document → Loader → Cleaner → Language detection → Chunker →
Tokeniser (NLTK stemming/stopwords) → Inverted Index + BM25 → Retriever
```

## Structure

```
src/
├── models/         keyword_model.py      — BM25Okapi wrapper
├── preprocessing/  preprocess.py         — clean, detect language, tokenise
├── indexing/       indexer.py            — build pipeline + inverted index
│                   bm25_store.py         — build/save/load BM25 model
├── retrieval/      retriever.py          — BM25 search + matched_terms
│                   keyword_adapter.py    — backend plug-in interface
├── evaluation/     evaluate.py           — precision@k, recall@k, MRR
└── utils/          chunker.py, loader.py, prompts.py
```

## Environment Variables (optional overrides)

| Variable               | Default               |
|------------------------|-----------------------|
| `KEYWORD_BM25_PATH`    | `keyword_bm25.pkl`    |
| `KEYWORD_INDEX_PATH`   | `keyword_index.pkl`   |
| `KEYWORD_CHUNKS_PATH`  | `keyword_chunks.pkl`  |
