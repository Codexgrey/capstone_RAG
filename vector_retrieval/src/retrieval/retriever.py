"""
src/retrieval/retriever.py
Similarity search logic.
Embeds a query and retrieves the top-k most similar chunks from the FAISS index.

Pipeline position:
    query → [retriever] → top-k chunks → return results
"""

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any


def l2_to_similarity(distance: float) -> float:
    """
    Convert an L2 distance to a similarity score in the range (0, 1].

    Args:
        distance: L2 distance value (non-negative).

    Returns:
        Similarity score — 1.0 is identical, approaches 0 for large distances.
    """
    return 1 / (1 + distance)


def retrieve(
    query: str,
    model: SentenceTransformer,
    index: faiss.IndexFlatL2,
    chunk_records: List[Dict[str, Any]],
    top_k: int = 3
) -> List[Dict[str, Any]]:
    """
    Embed the query and retrieve the top-k most similar chunks from the FAISS index.

    Args:
        query:         Natural language query string.
        model:         Loaded SentenceTransformer embedding model.
        index:         Populated FAISS index.
        chunk_records: List of chunk metadata dicts. May be in either:
                          - Research-pipeline schema: document_title, chunk_index,
                            word_count, source (from src/utils/chunker.py)
                          - Backend-produced schema (per shared_data/api_contracts):
                            source_name, page, start_char/end_char,
                            start_word_index/end_word_index/word_count, no
                            document_title / chunk_index / source.
                        Fields not present in the backend schema are derived
                        with sensible fallbacks so retrieval never raises
                        KeyError on either schema.
        top_k:         Number of top results to return.

    Returns:
        List of result dicts ordered by similarity (highest first), each containing:
            rank, document_id, document_title, source, chunk_id, chunk_index,
            word_count, distance, similarity, citation, text, metadata.

    Raises:
        ValueError: If the query is empty or whitespace.
    """
    if not query or not query.strip():
        raise ValueError('Query cannot be empty.')

    query_vector = model.encode([query], convert_to_numpy=True)
    query_vector = np.array(query_vector, dtype='float32')

    safe_top_k          = min(top_k, len(chunk_records))
    distances, indices  = index.search(query_vector, safe_top_k)

    results = []
    for rank, chunk_idx in enumerate(indices[0]):
        chunk           = chunk_records[int(chunk_idx)]
        distance_value  = float(distances[0][rank])
        similarity_value = float(l2_to_similarity(distance_value))

        source = chunk.get('source') or chunk.get('source_name') or 'Unknown'
        document_title = chunk.get('document_title') or _title_from_source(source)
        chunk_index = chunk.get('chunk_index')
        if chunk_index is None:
            chunk_index = _chunk_index_from_id(chunk.get('chunk_id', ''))
        word_count = chunk.get('word_count')
        if word_count is None:
            word_count = len(chunk.get('text', '').split())

        results.append({
            'rank':           rank + 1,
            'document_id':    chunk['document_id'],
            'document_title': document_title,
            'source':         source,
            'chunk_id':       chunk['chunk_id'],
            'chunk_index':    chunk_index,
            'word_count':     word_count,
            'distance':       distance_value,
            'similarity':     similarity_value,
            'score':          similarity_value,   # contract field — shared_data/schemas/retrieval_response.schema.json
            'citation':       f"[{document_title} | {chunk['chunk_id']}]",
            'text':           chunk['text'],
            'metadata':       chunk.get('metadata', {}),
        })

    return results


# ── Schema-fallback helpers ────────────────────────────────────────────────
# Mirrors hybrid_retrieval/src/retrieval/vector_retriever.py and
# bm25_retriever.py — kept local so this module stays self-contained.
def _title_from_source(source: str) -> str:
    """Derive a human-readable title from a source filename when
    document_title is absent (backend-produced chunks).

    Backend-produced source_name values look like:
        '47ce665d_Apple - CLaRa.txt' or 'doc_xyz_ocr.txt'
    Strip the 8-char hex upload prefix and OCR markers first, matching
    the cleanup already applied to 'source'/'source_name' in the backend
    bridges (see _clean() in backend/app/retrieval/*.py), so a derived
    title doesn't surface that internal naming to the user.
    """
    import os, re
    cleaned = re.sub(r'^[0-9a-f]{8}_', '', source)
    cleaned = cleaned.replace('_ocr.txt', '.pdf').replace('_ocr', '')
    name = os.path.splitext(cleaned)[0]
    return name.replace('_', ' ').replace('-', ' ').title()


def _chunk_index_from_id(chunk_id: str) -> int:
    """Best-effort extraction of a numeric chunk index from chunk_id
    (e.g. 'doc-001-chunk-0004' -> 3, or 'doc_abc123-chunk-7' -> 6)
    when chunk_index is absent."""
    import re
    match = re.search(r'(\d+)$', chunk_id or '')
    if match:
        return max(int(match.group(1)) - 1, 0)
    return 0
