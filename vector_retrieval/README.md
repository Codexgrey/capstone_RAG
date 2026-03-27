# vector_retrieval

Vector-based retrieval module for the Capstone RAG System.  
Uses `sentence-transformers` for dense embeddings and `FAISS` for similarity search.

---

## Structure

```
vector_retrieval/
│
├── src/
│   ├── models/
│   │   └── embedding_model.py     # loads & encodes with sentence-transformers
│   │
│   ├── indexing/
│   │   ├── indexer.py             # orchestrates load → chunk → embed → index
│   │   └── vector_store.py        # FAISS index setup
│   │
│   ├── retrieval/
│   │   └── retriever.py           # similarity search, returns top-k chunks
│   │
│   ├── evaluation/
│   │   └── evaluate.py            # precision@k, recall@k, MRR
│   │
│   ├── utils/
│   │   ├── loader.py              # reads document from disk
│   │   ├── chunker.py             # splits document into overlapping chunks
│   │   ├── prompts.py             # builds structured RAG prompt
│   │   └── response_printer.py   # formats and prints the final report
│   │
│   └── main.py                    # local entry point
│
├── tests/                         # unit tests
├── requirements.txt
└── README.md
```

---

## Pipeline

```
query → indexer (load → chunk → embed → index)
      → retriever (similarity search → top-k chunks)
      → prompt builder
      → generator (Groq LLM)
      → formatted response
```

---

## Setup

```bash
pip install -r requirements.txt
```

Add your Groq API key in `src/main.py`:
```python
GROQ_API_KEY = 'your_key_here'
```

Place your document at the path set in `DOCUMENT_SOURCE` in loader.py utils (default: `sample.txt`).

---

## Run

```bash
python -m src.main
```
