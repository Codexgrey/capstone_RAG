"""
src/retrieval/hybrid_adapter.py
Backend-facing adapter for the Hybrid Retrieval module (FAISS + BM25 + RRF).

Exposes two clean functions:
    ingest(file_paths, chunk_size, chunk_overlap) -> dict
    retrieve(query, top_k, rrf_k) -> dict

Ingestion is INCREMENTAL: new documents are appended to the existing FAISS
index and chunk records. Re-uploading a file with the same filename replaces
only that document's chunks (deduplication by source filename).

Contract references:
    shared_data/schemas/retrieval_request.schema.json
    shared_data/schemas/retrieval_response.schema.json
"""

import os
import time
import threading

import numpy as np
import faiss

from utils.loader               import load_document
from utils.chunker              import chunk_text_with_metadata
from models.embedding_model     import load_embedding_model
from indexing.vector_store      import load_index
from indexing.bm25_indexer      import build_inverted_index, build_bm25
from preprocessing.preprocess   import detect_language, tokenize_chunk
from retrieval.vector_retriever import retrieve as _vector_retrieve
from retrieval.bm25_retriever   import retrieve_bm25 as _bm25_retrieve
from retrieval.hybrid_retriever import reciprocal_rank_fusion

# ---------------------------------------------------------------------------
# persistence paths
# ---------------------------------------------------------------------------
VECTOR_INDEX_PATH  = os.environ.get('HYBRID_VECTOR_INDEX_PATH',  'hybrid_faiss_index.bin')
VECTOR_CHUNKS_PATH = os.environ.get('HYBRID_VECTOR_CHUNKS_PATH', 'hybrid_chunk_records.npy')
MODEL_NAME         = os.environ.get('HYBRID_MODEL_NAME',         'all-MiniLM-L6-v2')

# ---------------------------------------------------------------------------
# module-level state
# ---------------------------------------------------------------------------
_embedding_model = None
_faiss_index     = None
_chunk_records   = None
_bm25            = None
_inverted_index  = None
_nltk_lang       = None


def _get_model():
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = load_embedding_model(MODEL_NAME)
    return _embedding_model


def _load_state():
    global _embedding_model, _faiss_index, _chunk_records
    global _bm25, _inverted_index, _nltk_lang

    _get_model()
    _faiss_index, _chunk_records = load_index(VECTOR_INDEX_PATH, VECTOR_CHUNKS_PATH)

    sample, _nltk_lang  = _detect_lang(_chunk_records)
    tokenized           = [tokenize_chunk(c['text'], _nltk_lang) for c in _chunk_records]
    _inverted_index     = build_inverted_index(_chunk_records, tokenized)
    _bm25               = build_bm25(tokenized)


def _reset_state():
    global _faiss_index, _chunk_records, _bm25, _inverted_index, _nltk_lang
    _faiss_index = _chunk_records = _bm25 = _inverted_index = _nltk_lang = None


def _detect_lang(chunks):
    sample = ' '.join(c['text'] for c in chunks[:20])
    _, lang = detect_language(sample)
    return sample, lang


# ---------------------------------------------------------------------------
# public interface
# ---------------------------------------------------------------------------

def ingest(
    file_paths:    list,
    chunk_size:    int = 150,
    chunk_overlap: int = 30,
) -> dict:
    """
    Incrementally ingest one or more documents into the hybrid index.

    New documents are APPENDED. Re-uploading a file with the same filename
    replaces only that document's chunks.

    Returns:
        {"status": "ok"|"error", "documents_ingested": int,
         "total_chunks": int, "latency_ms": float, "error": str|None}
    """
    if not file_paths:
        raise ValueError('file_paths must contain at least one file path.')

    start = time.perf_counter()
    model = _get_model()

    try:
        # ── Load existing index if present ────────────────────────────────
        if os.path.exists(VECTOR_INDEX_PATH) and os.path.exists(VECTOR_CHUNKS_PATH):
            existing_index, existing_chunks = load_index(VECTOR_INDEX_PATH, VECTOR_CHUNKS_PATH)
        else:
            existing_index  = None
            existing_chunks = []

        existing_sources = {c['source'] for c in existing_chunks}
        existing_doc_count = len({c['document_id'] for c in existing_chunks})

        # ── Load and chunk new documents ──────────────────────────────────
        new_chunks = []
        new_sources = set()

        for path in file_paths:
            filename  = os.path.basename(path)
            doc_index = existing_doc_count + len({c['document_id'] for c in new_chunks})
            doc_id    = f'doc-{doc_index + 1:03d}'
            doc_title = (
                os.path.splitext(filename)[0]
                .replace('_', ' ').replace('-', ' ').title()
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
            new_chunks.extend(chunks)
            new_sources.add(filename)

        if not new_chunks:
            return {'status': 'error', 'documents_ingested': 0, 'total_chunks': 0,
                    'latency_ms': 0.0, 'error': 'No chunks produced from provided files.'}

        # ── Drop replaced sources from existing set ───────────────────────
        kept_chunks = [c for c in existing_chunks if c['source'] not in new_sources]

        # ── Embed new chunks ──────────────────────────────────────────────
        new_texts = [c['text'] for c in new_chunks]
        new_vecs  = model.encode(new_texts, convert_to_numpy=True, show_progress_bar=False)
        new_vecs  = np.array(new_vecs, dtype='float32')

        # ── Merge vectors (reconstruct kept from existing FAISS) ──────────
        if kept_chunks and existing_index is not None:
            kept_sources   = {c['source'] for c in kept_chunks}
            kept_positions = [i for i, c in enumerate(existing_chunks)
                              if c['source'] in kept_sources]
            kept_vecs      = np.vstack([
                existing_index.reconstruct(pos) for pos in kept_positions
            ]).astype('float32')
            all_vecs   = np.vstack([kept_vecs, new_vecs])
        else:
            all_vecs   = new_vecs
            kept_chunks = []

        all_chunks = kept_chunks + new_chunks

        # ── Build and save new FAISS index ────────────────────────────────
        dim   = all_vecs.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(all_vecs)

        faiss.write_index(index, VECTOR_INDEX_PATH)
        np.save(VECTOR_CHUNKS_PATH, np.array(all_chunks, dtype=object))

        with _lock:
            _reset_state()  # force reload on next retrieve()

        latency_ms = round((time.perf_counter() - start) * 1000, 2)
        print(f'  ✅ Hybrid index: {len(new_chunks)} new chunks, '
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


def retrieve(
    query: str,
    top_k: int = 3,
    rrf_k: int = 60,
) -> dict:
    """
    Retrieve top-k chunks using hybrid RRF (FAISS + BM25).

    Loads model and indexes on first call; reuses state across subsequent calls.
    """
    if not query or not query.strip():
        raise ValueError('query cannot be empty.')

    if _faiss_index is None or _chunk_records is None:
        _load_state()

    start = time.perf_counter()

    candidate_k    = max(top_k * 2, top_k)
    vector_results = _vector_retrieve(
        query, _embedding_model, _faiss_index, _chunk_records, top_k=candidate_k,
    )
    bm25_results = _bm25_retrieve(
        query, _bm25, _chunk_records, _inverted_index,
        nltk_lang=_nltk_lang, top_k=candidate_k,
    )
    hybrid_results = reciprocal_rank_fusion(
        bm25_results=bm25_results, vector_results=vector_results,
        k=rrf_k, top_k=top_k,
    )

    latency_ms = round((time.perf_counter() - start) * 1000, 2)

    return {
        'query':      query,
        'method':     'hybrid',
        'results':    hybrid_results,
        'latency_ms': latency_ms,
    }
