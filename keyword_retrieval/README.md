# Keyword Retrieval (BM25 + Inverted Index) 

Keyword-based retrieval module of the `Capstone_RAG` project. Given a user
query, the system:

1. Ingests a document (PDF / Word / TXT / web page — TXT implemented, others
   extendable in `src/utils/loader.py`).
2. Detects the document language (English, French, Turkish, German, Spanish,
   Italian, Portuguese, Arabic, ...).
3. Cleans, chunks (300–500 word blocks with overlap), and tokenises the text
   (lowercase → stopword removal → stemming).
4. Builds a positional inverted index and a BM25 store over the same tokens.
5. Normalises the user query with a small LLM, then BM25-ranks the inverted
   index to return the top-K chunks.
6. Builds a grounded prompt and asks a larger LLM (Groq) to answer using
   **only** that context.

Pipeline diagram:

```
Document → Loader → Cleaner → Chunker → Tokenizer → Inverted Index (BM25)
→ Query → Query Normaliser (LLM) → BM25 Ranker → Retriever
→ Prompt Builder → Generator → Final Response
```

## Folder Structure

```
keyword_retrieval/
│
├── src/
│   ├── models/            # LLM wrappers (query normaliser + generator)
│   │   └── keyword_model.py
│   │
│   ├── preprocessing/     # Language detection, cleaning, tokenisation
│   │   └── preprocess.py
│   │
│   ├── indexing/          # Inverted index + BM25 store
│   │   ├── indexer.py
│   │   └── bm25_store.py
│   │
│   ├── retrieval/         # BM25 ranking + top-K retrieval
│   │   └── retriever.py
│   │
│   ├── evaluation/        # Precision@k, Recall@k, MRR
│   │   └── evaluate.py
│   │
│   ├── utils/             # Loader, chunker, prompt builder
│   │   ├── loader.py
│   │   ├── chunker.py
│   │   └── prompts.py
│   │
│   └── main.py            # Local entry point (argparse)
│
├── tests/
│   └── test_pipeline.py   # Offline unit tests (no API key required)
│
├── data/
│   └── sample.txt         # Example document for manual testing
│
├── keyword_retrieval.ipynb  # Interactive pipeline notebook (MVP)
├── requirements.txt
└── README.md
```

## Quick Start

```bash
# 1. Install deps
pip install -r requirements.txt

# 2. Set the Groq API key
export GROQ_API_KEY='your-groq-key-here'

# 3. Run the full pipeline from the CLI
python -m src.main --doc data/sample.txt \
    --query "How does the inverted index work in keyword search systems?"

# 4. Or run the interactive notebook
jupyter notebook keyword_retrieval.ipynb
```

## Notebook Overview (`keyword_retrieval.ipynb`)

The notebook mirrors every stage of the pipeline so each can be tested,
debugged, and understood before being moved into production:

1.  Environment Setup
2.  Test Document Setup
3.  Loader (`utils/loader.py`)
4.  Language Detection (`preprocessing/preprocess.py`)
5.  Text Cleaning (`preprocessing/preprocess.py`)
6.  Chunker (`utils/chunker.py`)
7.  Tokenizer (`preprocessing/preprocess.py`)
8.  Build Inverted Index (`indexing/indexer.py` + `indexing/bm25_store.py`) **← heart of the system**
9.  Index Inspection
10. Query Normaliser — small LLM (`models/keyword_model.py`)
11. BM25 Ranking + Retriever (`retrieval/retriever.py`)
12. Retrieval Debug View
13. Prompt Builder (`utils/prompts.py`)
14. Generator (Groq API)
15. End-to-End Pipeline Test
16. Summary

## Tests

```bash
python -m unittest tests.test_pipeline
```

The offline tests do not call the Groq API and validate chunking,
tokenisation, inverted-index construction, BM25 retrieval, and language
detection.

## Notes

* The inverted index stores term frequency, document frequency, and token
  positions. Positional data enables phrase search (e.g. "New York" instead
  of just "New" and "York") on top of the boolean / BM25 lookups.
* BM25 scoring accounts for term frequency, rarity, and chunk length.
* Language detection is automatic — stopwords adapt per document language.
* The generator is instructed to reply with `"I don't have enough
  information."` when the retrieved context does not support the answer.