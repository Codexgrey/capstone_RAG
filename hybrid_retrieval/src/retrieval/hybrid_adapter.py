"""
src/retrieval/hybrid_adapter.py
Backend-facing adapter for the Hybrid Retrieval module.

This is the integration boundary between this module and the backend.
It exposes two clean functions that together form the complete plug-in interface:

    ingest(file_paths, chunk_size, chunk_overlap) -> dict
    retrieve(query, top_k, rrf_k) -> dict

The backend calls ingest() after a user uploads documents, then calls retrieve()
to answer queries against the indexed content. Neither function exposes any
internal implementation details (FAISS, BM25, RRF fusion, chunker, etc).

Internally, retrieve() runs both vector and keyword retrieval in parallel, then
fuses the ranked lists using Reciprocal Rank Fusion (RRF) before returning the
merged top-k results. The caller receives a single unified result list — the
hybrid nature is transparent.

The adapter manages all module-level state internally. Models and indexes are
loaded once on the first retrieve() call and reused across all subsequent calls
for efficiency. After ingest() rebuilds the indexes, the adapter reloads them
automatically on the next retrieve() call.

Contract references:
    shared_data/schemas/retrieval_request.schema.json
    shared_data/schemas/retrieval_response.schema.json
"""

import os
import time

from src.utils.loader                  import load_document
from src.utils.chunker                 import chunk_text_with_metadata
from src.models.embedding_model        import load_embedding_model
from src.indexing.vector_store         import build_and_save_index, load_index
from src.indexing.bm25_indexer         import build_inverted_index, build_bm25
from src.preprocessing.preprocess     import detect_language, tokenize_chunk
from src.retrieval.vector_retriever    import retrieve as _vector_retrieve
from src.retrieval.bm25_retriever      import retrieve_bm25 as _bm25_retrieve
from src.retrieval.hybrid_retriever    import reciprocal_rank_fusion


# ---------------------------------------------------------------------------
# persistence paths — configurable via environment variables
# ---------------------------------------------------------------------------
VECTOR_INDEX_PATH  = os.environ.get('HYBRID_VECTOR_INDEX_PATH',  'hybrid_faiss_index.bin')
VECTOR_CHUNKS_PATH = os.environ.get('HYBRID_VECTOR_CHUNKS_PATH', 'hybrid_chunk_records.npy')
MODEL_NAME         = os.environ.get('HYBRID_MODEL_NAME',         'all-MiniLM-L6-v2')


# ---------------------------------------------------------------------------
# module-level state — loaded once, reused across calls
# ---------------------------------------------------------------------------
_embedding_model = None   # SentenceTransformer — preserved across re-ingests
_faiss_index     = None   # FAISS flat index
_chunk_records   = None   # list[dict] — shared by both retrievers
_bm25            = None   # BM25Okapi object
_inverted_index  = None   # dict — term → postings
_nltk_lang       = None   # detected corpus language string (e.g. 'english')


def _load_state() -> None:
    """
    Load the embedding model, FAISS index, and chunk records into module-level
    state. The BM25 model and inverted index are rebuilt from the loaded chunks
    because they are not persisted to disk separately.

    Called lazily on the first retrieve() call and after ingest() rebuilds the
    indexes.

    Raises:
        FileNotFoundError: If the FAISS index or chunk records file do not exist
                           (ingest has not yet been called).
        RuntimeError:      If the embedding model fails to load.
    """
    global _embedding_model, _faiss_index, _chunk_records
    global _bm25, _inverted_index, _nltk_lang

    # Load embedding model once — survives re-ingests
    if _embedding_model is None:
        _embedding_model = load_embedding_model(MODEL_NAME)

    # Load persisted FAISS index and chunk records
    _faiss_index, _chunk_records = load_index(VECTOR_INDEX_PATH, VECTOR_CHUNKS_PATH)

    # Rebuild BM25 and inverted index in-memory from chunk records
    full_text          = ' '.join(c['text'] for c in _chunk_records)
    _, _nltk_lang      = detect_language(full_text)
    tokenized_chunks   = [tokenize_chunk(c['text'], _nltk_lang) for c in _chunk_records]
    _inverted_index    = build_inverted_index(_chunk_records, tokenized_chunks)
    _bm25              = build_bm25(tokenized_chunks)


def _reset_state() -> None:
    """
    Clear index state so retrieve() reloads from the new index on next call.
    The embedding model is preserved — no need to reload after re-indexing.
    """
    global _faiss_index, _chunk_records, _bm25, _inverted_index, _nltk_lang
    _faiss_index    = None
    _chunk_records  = None
    _bm25           = None
    _inverted_index = None
    _nltk_lang      = None


# ---------------------------------------------------------------------------
# public interface
# ---------------------------------------------------------------------------

def ingest(
    file_paths:    list,
    chunk_size:    int = 150,
    chunk_overlap: int = 30,
) -> dict:
    """
    Ingest one or more documents into the hybrid index.

    Called by the backend after a user uploads files. Runs the full pipeline:
    load → chunk → embed → FAISS index → persist. The BM25 index is rebuilt
    in-memory from the persisted chunks on the next retrieve() call.

    Args:
        file_paths:    List of absolute or relative file paths to ingest.
                       Supported types: .txt, .md, .pdf, .docx
        chunk_size:    Number of words per chunk. Default: 150.
        chunk_overlap: Number of overlapping words between adjacent chunks.
                       Default: 30.

    Returns:
        {
            "status":             "ok" | "error",
            "documents_ingested": int,
            "total_chunks":       int,
            "latency_ms":         float,
            "error":              str | None
        }

    Raises:
        ValueError: If file_paths is empty.
    """
    if not file_paths:
        raise ValueError('file_paths must contain at least one file path.')

    start = time.perf_counter()

    try:
        import numpy as np

        # Ensure the embedding model is available for this run
        global _embedding_model
        if _embedding_model is None:
            _embedding_model = load_embedding_model(MODEL_NAME)

        # ── 1. Load and chunk all documents ──────────────────────────
        all_chunk_records = []

        for doc_index, path in enumerate(file_paths):
            doc_id    = f'doc-{doc_index + 1:03d}'
            doc_title = (
                os.path.splitext(os.path.basename(path))[0]
                .replace('_', ' ')
                .replace('-', ' ')
                .title()
            )
            text, file_metadata = load_document(path)
            chunks = chunk_text_with_metadata(
                text,
                chunk_size     = chunk_size,
                overlap        = chunk_overlap,
                document_title = doc_title,
                source         = file_metadata['file_name'],
                document_id    = doc_id,
                file_metadata  = file_metadata,
            )
            all_chunk_records.extend(chunks)

        # ── 2. Build and persist FAISS index ─────────────────────────
        chunk_texts = [c['text'] for c in all_chunk_records]
        embeddings  = _embedding_model.encode(
            chunk_texts,
            convert_to_numpy  = True,
            show_progress_bar = False,
        )
        embeddings = np.array(embeddings, dtype='float32')

        build_and_save_index(
            embeddings,
            all_chunk_records,
            VECTOR_INDEX_PATH,
            VECTOR_CHUNKS_PATH,
        )

        # ── 3. Signal retrieve() to reload on next call ───────────────
        _reset_state()

        latency_ms = round((time.perf_counter() - start) * 1000, 2)

        return {
            'status':             'ok',
            'documents_ingested': len(file_paths),
            'total_chunks':       len(all_chunk_records),
            'latency_ms':         latency_ms,
            'error':              None,
        }

    except Exception as e:
        latency_ms = round((time.perf_counter() - start) * 1000, 2)
        return {
            'status':             'error',
            'documents_ingested': 0,
            'total_chunks':       0,
            'latency_ms':         latency_ms,
            'error':              str(e),
        }


def retrieve(
    query:   str,
    top_k:   int = 3,
    rrf_k:   int = 60,
) -> dict:
    """
    Retrieve the top-k most relevant chunks for a query using hybrid
    Reciprocal Rank Fusion (RRF) over vector and keyword results.

    Loads the model and indexes on first call; subsequent calls reuse state.
    The indexes searched are always the most recently ingested ones.

    Internally:
        1. Vector retrieval — FAISS similarity search (all-MiniLM-L6-v2)
        2. Keyword retrieval — BM25 scoring over tokenised chunks
        3. RRF fusion — combines both ranked lists into a single ranking

    Each result record carries the original vector similarity score,
    the BM25 score, the RRF score, and a 'retrieval' field indicating
    whether the chunk was found by keyword only, vector only, or both.

    Args:
        query:  Natural language question string.
        top_k:  Number of top chunks to return. Default: 3.
        rrf_k:  RRF smoothing constant. Higher values reduce the impact of
                top-ranked results. Default: 60 (standard literature value).

    Returns:
        Dict matching retrieval_response.schema.json:
        {
            "query":      str,
            "method":     "hybrid",
            "results":    list[dict],
            "latency_ms": float
        }

        Each result dict contains:
            chunk_id, document_id, document_title, source, chunk_index,
            word_count, text, citation, rank, rrf_score, bm25_score,
            similarity, retrieval (source: 'keyword only' | 'vector only'
            | 'BOTH (keyword + vector)')

    Raises:
        ValueError:        If query is empty.
        FileNotFoundError: If index files are not found (ingest not yet called).
    """
    if not query or not query.strip():
        raise ValueError('query cannot be empty.')

    # Lazy-load all state on first call or after a re-ingest
    if _faiss_index is None or _chunk_records is None:
        _load_state()

    start = time.perf_counter()

    # ── 1. Vector retrieval ───────────────────────────────────────────
    # Fetch a wider candidate pool (top_k * 2) so RRF has more signal to
    # work with before collapsing to the final top_k.
    candidate_k     = max(top_k * 2, top_k)
    vector_results  = _vector_retrieve(
        query,
        _embedding_model,
        _faiss_index,
        _chunk_records,
        top_k = candidate_k,
    )

    # ── 2. Keyword retrieval (BM25) ───────────────────────────────────
    # BM25 does not use LLM-based query normalisation here; the raw query
    # is tokenised with the same pipeline used at index time, keeping the
    # adapter stateless and free of external API dependencies at query time.
    bm25_results = _bm25_retrieve(
        query,
        _bm25,
        _chunk_records,
        _inverted_index,
        nltk_lang = _nltk_lang,
        top_k     = candidate_k,
    )

    # ── 3. Reciprocal Rank Fusion ─────────────────────────────────────
    hybrid_results = reciprocal_rank_fusion(
        bm25_results   = bm25_results,
        vector_results = vector_results,
        k              = rrf_k,
        top_k          = top_k,
    )

    latency_ms = round((time.perf_counter() - start) * 1000, 2)

    return {
        'query':      query,
        'method':     'hybrid',
        'results':    hybrid_results,
        'latency_ms': latency_ms,
    }
