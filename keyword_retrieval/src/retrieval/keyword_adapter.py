"""
src/retrieval/keyword_adapter.py
Backend-facing adapter for the Keyword Retrieval module.

Exposes two clean functions:
    ingest(file_paths, chunk_size, chunk_overlap) -> dict
    retrieve(query, top_k) -> dict

Ingestion is INCREMENTAL: new documents are appended to the existing BM25
index and chunk records. Re-uploading a file with the same filename replaces
only that document's chunks (deduplication by source filename).

Contract references:
    shared_data/schemas/retrieval_request.schema.json
    shared_data/schemas/retrieval_response.schema.json
"""

import os
import pickle
import time
import threading

from indexing.indexer    import build_pipeline
from indexing.bm25_store import load_bm25, build_bm25, save_bm25
from retrieval.retriever import retrieve as _retrieve

# ---------------------------------------------------------------------------
# persistence paths
# ---------------------------------------------------------------------------
INDEX_PATH  = os.environ.get('KEYWORD_INDEX_PATH',  'keyword_index.pkl')
BM25_PATH   = os.environ.get('KEYWORD_BM25_PATH',   'keyword_bm25.pkl')
CHUNKS_PATH = os.environ.get('KEYWORD_CHUNKS_PATH', 'keyword_chunks.pkl')

# ---------------------------------------------------------------------------
# module-level state
# ---------------------------------------------------------------------------
_bm25          = None
_index         = None
_chunk_records = None
_lock          = threading.RLock()  # serialise state resets


def _load_state():
    global _bm25, _index, _chunk_records
    _bm25, _index, _chunk_records = load_bm25(BM25_PATH, INDEX_PATH, CHUNKS_PATH)


def _reset_state():
    global _bm25, _index, _chunk_records
    _bm25 = _index = _chunk_records = None
_lock          = threading.RLock()  # serialise state resets


# ---------------------------------------------------------------------------
# public interface
# ---------------------------------------------------------------------------

def ingest(
    file_paths:    list,
    chunk_size:    int = 300,
    chunk_overlap: int = 50,
) -> dict:
    """
    Incrementally ingest one or more documents into the keyword index.

    New documents are APPENDED. Re-uploading a file with the same filename
    replaces only that document's chunks.

    Returns:
        {"status": "ok"|"error", "documents_ingested": int,
         "total_chunks": int, "latency_ms": float, "error": str|None}
    """
    if not file_paths:
        raise ValueError('file_paths must contain at least one file path.')

    start = time.perf_counter()

    try:
        # ── Load existing chunks if present ───────────────────────────────
        if os.path.exists(CHUNKS_PATH):
            with open(CHUNKS_PATH, 'rb') as f:
                existing_chunks = pickle.load(f)
        else:
            existing_chunks = []

        existing_sources = {c['source'] for c in existing_chunks}

        # ── Build new chunks from provided files ──────────────────────────
        new_chunk_records, _ = build_pipeline(
            file_paths    = file_paths,
            chunk_size    = chunk_size,
            chunk_overlap = chunk_overlap,
            index_path    = '_tmp_index.pkl',   # temp — merged below
            bm25_path     = '_tmp_bm25.pkl',
            chunks_path   = '_tmp_chunks.pkl',
        )

        # ── Determine which sources to replace ────────────────────────────
        new_sources = {c['source'] for c in new_chunk_records}
        kept_chunks = [c for c in existing_chunks
                       if c['source'] not in new_sources]

        all_chunks = kept_chunks + new_chunk_records

        # ── Rebuild BM25 + inverted index over ALL chunks ─────────────────
        from preprocessing.preprocess import tokenize_chunk, detect_language
        from indexing.indexer import build_inverted_index

        # Detect language from combined text sample
        sample          = ' '.join(c['text'] for c in all_chunks[:20])
        _, nltk_lang    = detect_language(sample)
        tokenized       = [tokenize_chunk(c['text'], nltk_lang) for c in all_chunks]

        new_index       = build_inverted_index(all_chunks, tokenized)
        new_bm25        = build_bm25(tokenized)

        # ── Persist merged state ──────────────────────────────────────────
        with open(INDEX_PATH,  'wb') as f: pickle.dump(new_index,  f)
        with open(CHUNKS_PATH, 'wb') as f: pickle.dump(all_chunks, f)
        save_bm25(new_bm25, BM25_PATH)

        # Clean up temp files
        for tmp in ('_tmp_index.pkl', '_tmp_bm25.pkl', '_tmp_chunks.pkl'):
            if os.path.exists(tmp):
                os.remove(tmp)

        with _lock:
            _reset_state()  # force reload on next retrieve()

        latency_ms = round((time.perf_counter() - start) * 1000, 2)
        print(f'  ✅ Keyword index: {len(new_chunk_records)} new chunks, '
              f'{len(kept_chunks)} kept, {len(all_chunks)} total')

        return {
            'status':             'ok',
            'documents_ingested': len(file_paths),
            'total_chunks':       len(all_chunks),
            'latency_ms':         latency_ms,
            'error':              None,
        }

    except Exception as e:
        for tmp in ('_tmp_index.pkl', '_tmp_bm25.pkl', '_tmp_chunks.pkl'):
            if os.path.exists(tmp):
                os.remove(tmp)
        latency_ms = round((time.perf_counter() - start) * 1000, 2)
        return {'status': 'error', 'documents_ingested': 0, 'total_chunks': 0,
                'latency_ms': latency_ms, 'error': str(e)}


def retrieve(query: str, top_k: int = 3) -> dict:
    """
    Retrieve top-k chunks using BM25 keyword search.

    Loads state on first call; reuses across subsequent calls.
    """
    if not query or not query.strip():
        raise ValueError('query cannot be empty.')

    if _bm25 is None:
        _load_state()

    start      = time.perf_counter()
    results    = _retrieve(query, _bm25, _chunk_records, _index, top_k=top_k)
    latency_ms = round((time.perf_counter() - start) * 1000, 2)

    return {
        'query':      query,
        'method':     'keyword',
        'results':    results,
        'latency_ms': latency_ms,
    }
