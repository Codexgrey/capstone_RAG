"""
src/retrieval/vector_adapter.py
Backend-facing adapter for the Vector Retrieval module.

Exposes two clean functions:
    ingest(file_paths, chunk_size, chunk_overlap) -> dict
    retrieve(query, top_k) -> dict

Ingestion is INCREMENTAL: new documents are appended to the existing
FAISS index and chunk records rather than replacing them. Re-uploading
a file that already exists (same filename) replaces only that document's
chunks (deduplication by source filename).

Contract references:
    shared_data/schemas/retrieval_request.schema.json
    shared_data/schemas/retrieval_response.schema.json
"""

import os
import time
import threading

import numpy as np
import faiss

from src.utils.loader import load_document
from src.utils.chunker import chunk_text_with_metadata
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
_lock          = threading.RLock()  # serialise state resets


def _load_state():
    global _index, _chunk_records
    if _model is None:
        _get_model()
    _index, _chunk_records = load_index(INDEX_PATH, CHUNKS_PATH)


# ---------------------------------------------------------------------------
# public interface
# ---------------------------------------------------------------------------

def ingest(
    file_paths:    list,
    chunk_size:    int = 300,
    chunk_overlap: int = 50,
) -> dict:
    """
    Incrementally ingest one or more documents into the vector index.

    New documents are APPENDED to the existing index. Re-uploading a file
    with the same filename replaces only that document's chunks (deduplication
    by source filename). The first call creates the index from scratch.

    Returns:
        {"status": "ok"|"error", "documents_ingested": int,
         "total_chunks": int, "latency_ms": float, "error": str|None}
    """
    if not file_paths:
        raise ValueError('file_paths must contain at least one file path.')

    start  = time.perf_counter()
    model  = _get_model()

    try:
        # ── Load existing index if present ────────────────────────────────
        if os.path.exists(INDEX_PATH) and os.path.exists(CHUNKS_PATH):
            existing_index, existing_chunks = load_index(INDEX_PATH, CHUNKS_PATH)
        else:
            existing_index  = None
            existing_chunks = []

        # ── Determine doc-id offset (so new IDs don't collide) ────────────
        existing_sources = {c['source'] for c in existing_chunks}
        existing_count   = len({c['document_id'] for c in existing_chunks})

        # ── Load and chunk new documents ──────────────────────────────────
        new_chunks = []
        sources_to_replace = set()

        for path in file_paths:
            filename  = os.path.basename(path)
            doc_index = existing_count + len({c['document_id'] for c in new_chunks})
            doc_id    = f'doc-{doc_index + 1:03d}'
            doc_title = (
                os.path.splitext(filename)[0]
                .replace('_', ' ').replace('-', ' ').title()
            )

            text, metadata = load_document(path)
            chunks = chunk_text_with_metadata(
                text,
                chunk_size     = chunk_size,
                chunk_overlap  = chunk_overlap,
                document_title = doc_title,
                source         = metadata['file_name'],
                document_id    = doc_id,
                file_metadata  = metadata,
            )
            new_chunks.extend(chunks)

            # Mark this source for replacement if it already exists
            if filename in existing_sources:
                sources_to_replace.add(filename)

        if not new_chunks:
            return {'status': 'error', 'documents_ingested': 0, 'total_chunks': 0,
                    'latency_ms': 0.0, 'error': 'No chunks produced from provided files.'}

        # ── Remove replaced sources from existing chunks ──────────────────
        kept_chunks = [c for c in existing_chunks
                       if c['source'] not in sources_to_replace]

        # ── Embed new chunks ──────────────────────────────────────────────
        new_texts      = [c['text'] for c in new_chunks]
        new_embeddings = model.encode(new_texts, convert_to_numpy=True, show_progress_bar=False)
        new_embeddings = np.array(new_embeddings, dtype='float32')

        # ── Merge: re-embed kept chunks only if index is being rebuilt ────
        if kept_chunks and existing_index is not None:
            # Re-extract kept vectors from the existing FAISS index by position
            kept_sources   = {c['source'] for c in kept_chunks}
            kept_positions = [i for i, c in enumerate(existing_chunks)
                              if c['source'] in kept_sources]
            kept_vecs      = np.vstack([
                existing_index.reconstruct(pos) for pos in kept_positions
            ]).astype('float32')
            all_embeddings = np.vstack([kept_vecs, new_embeddings])
        else:
            all_embeddings = new_embeddings
            kept_chunks    = []

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
              f'{len(kept_chunks)} kept, {len(all_chunks)} total')

        return {
            'status':             'ok',
            'documents_ingested': len(file_paths),
            'total_chunks':       len(all_chunks),
            'latency_ms':         latency_ms,
            'error':              None,
        }

    except Exception as e:
        latency_ms = round((time.perf_counter() - start) * 1000, 2)
        return {'status': 'error', 'documents_ingested': 0, 'total_chunks': 0,
                'latency_ms': latency_ms, 'error': str(e)}


def retrieve(query: str, top_k: int = 3) -> dict:
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
