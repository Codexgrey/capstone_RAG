"""
vector_retrieval/src/retrieval/vector_adapter.py
Backend-facing adapter for the Vector Retrieval module.

Exposes two clean functions, matching the shared adapter contract used by
keyword_retrieval/src/retrieval/keyword_adapter.py and
hybrid_retrieval/src/retrieval/hybrid_adapter.py:

    ingest(chunks, document_id) -> dict
    retrieve(query, top_k)      -> dict

The backend passes pre-parsed, pre-chunked chunk dicts (already in the
shared schema produced by backend/app/ingestion/chunker.py — fixed
200-word / 40-word-overlap chunks). This adapter does NOT re-chunk or
re-load files; it embeds the chunk text it is given and indexes it in
FAISS, the same way keyword_adapter.ingest() builds a BM25 index over
the same chunks.

Ingestion is INCREMENTAL: new documents are appended to the existing
FAISS index and chunk records rather than replacing them. Re-uploading
a document (same document_id) replaces only that document's chunks.

NOTE: vector_retrieval/src/main.py (the standalone CLI / research
pipeline) is a separate execution path that uses its own module-level
chunker (src/utils/chunker.py) and loader (src/utils/loader.py) directly
via src/indexing/indexer.py — it does not call this adapter and is
unaffected by this change.

Contract references:
    shared_data/schemas/retrieval_request.schema.json
    shared_data/schemas/retrieval_response.schema.json
"""

import os
import time
import threading

import numpy as np
import faiss

from src.models.embedding_model import load_embedding_model
from src.indexing.vector_store import load_index
from src.retrieval.retriever import retrieve as _retrieve

# ---------------------------------------------------------------------------
# persistence paths
# ---------------------------------------------------------------------------
INDEX_PATH  = os.environ.get('VECTOR_INDEX_PATH',  'faiss_index.bin')
CHUNKS_PATH = os.environ.get('VECTOR_CHUNKS_PATH', 'chunk_records.npy')
MODEL_NAME  = os.environ.get('VECTOR_MODEL_NAME',  'all-MiniLM-L6-v2')

# ---------------------------------------------------------------------------
# module-level state
# ---------------------------------------------------------------------------
_model         = None
_index         = None
_chunk_records = None
_lock          = threading.RLock()  # serialise state resets


def _get_model():
    global _model
    if _model is None:
        _model = load_embedding_model(MODEL_NAME)
    return _model


def _reset_state():
    global _index, _chunk_records
    _index         = None
    _chunk_records = None


def _load_state():
    global _index, _chunk_records
    if _model is None:
        _get_model()
    _index, _chunk_records = load_index(INDEX_PATH, CHUNKS_PATH)


# ---------------------------------------------------------------------------
# public interface
# ---------------------------------------------------------------------------

def ingest(chunks: list, document_id: str) -> dict:
    """
    Incrementally ingest pre-chunked text into the vector index.

    New chunks are APPENDED to the existing index. Re-ingesting a
    document (same document_id) replaces only that document's prior
    chunks. The first call creates the index from scratch.

    The backend chunker already produces chunks in the shared schema:
        {chunk_id, document_id, source_name, text, page,
         start_char, end_char, start_word_index, end_word_index, word_count}

    Returns:
        {"status": "ok"|"error", "documents_ingested": int,
         "total_chunks": int, "latency_ms": float, "error": str|None}
    """
    if not chunks:
        return {'status': 'error', 'documents_ingested': 0, 'total_chunks': 0,
                'latency_ms': 0.0, 'error': 'chunks must contain at least one chunk dict.'}

    start = time.perf_counter()
    model = _get_model()

    try:
        # ── Load existing index if present ────────────────────────────────
        if os.path.exists(INDEX_PATH) and os.path.exists(CHUNKS_PATH):
            existing_index, existing_chunks = load_index(INDEX_PATH, CHUNKS_PATH)
        else:
            existing_index  = None
            existing_chunks = []

        # ── Remove prior chunks for this document (re-ingest safety) ──────
        kept_chunks = [c for c in existing_chunks
                       if c.get('document_id') != document_id]
        is_reingest = len(kept_chunks) != len(existing_chunks)

        new_chunks = chunks

        # ── Embed new chunks ──────────────────────────────────────────────
        new_texts      = [c['text'] for c in new_chunks]
        new_embeddings = model.encode(new_texts, convert_to_numpy=True, show_progress_bar=False)
        new_embeddings = np.array(new_embeddings, dtype='float32')

        # ── Merge: re-extract kept vectors only if index is being rebuilt ─
        if kept_chunks and existing_index is not None:
            kept_positions = [i for i, c in enumerate(existing_chunks)
                               if c.get('document_id') != document_id]
            kept_vecs      = np.vstack([
                existing_index.reconstruct(pos) for pos in kept_positions
            ]).astype('float32')
            all_embeddings = np.vstack([kept_vecs, new_embeddings])
        else:
            all_embeddings = new_embeddings
            kept_chunks    = [] if not kept_chunks else kept_chunks

        all_chunks = kept_chunks + new_chunks

        # ── Build and save new index ──────────────────────────────────────
        dim   = all_embeddings.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(all_embeddings)

        faiss.write_index(index, INDEX_PATH)
        np.save(CHUNKS_PATH, np.array(all_chunks, dtype=object))

        with _lock:
            _reset_state()  # force reload on next retrieve()

        latency_ms = round((time.perf_counter() - start) * 1000, 2)
        print(f'  ✅ Vector index: {len(new_chunks)} new chunks, '
              f'{len(kept_chunks)} kept, {len(all_chunks)} total'
              + (' (re-ingest)' if is_reingest else ''))

        return {
            'status':             'ok',
            'documents_ingested': 1,
            'total_chunks':       len(all_chunks),
            'latency_ms':         latency_ms,
            'error':              None,
        }

    except Exception as e:
        latency_ms = round((time.perf_counter() - start) * 1000, 2)
        return {'status': 'error', 'documents_ingested': 0, 'total_chunks': 0,
                'latency_ms': latency_ms, 'error': str(e)}


def retrieve(query: str, top_k: int = 5) -> dict:
    """
    Retrieve the top-k most similar chunks for a query.

    Loads model + index on first call; reuses state across subsequent calls.
    """
    if not query or not query.strip():
        raise ValueError('query cannot be empty.')

    if _index is None or _chunk_records is None:
        _load_state()

    start      = time.perf_counter()
    results    = _retrieve(query, _model, _index, _chunk_records, top_k=top_k)
    latency_ms = round((time.perf_counter() - start) * 1000, 2)

    return {
        'query':      query,
        'method':     'vector',
        'results':    results,
        'latency_ms': latency_ms,
    }
