import numpy as np

def l2_to_similarity(distance):
    """
    Converts L2 distance to a similarity score between 0 and 1.
    Lower distance = higher similarity.
    """
    return 1 / (1 + distance)


def retrieve(query, model, index, chunk_records, top_k=3):
    """
    Embeds the query, searches the FAISS index,
    and returns the top-k most similar chunks.

    chunk_records may contain two different schemas:
      - Research-pipeline chunks: document_title, chunk_index, word_count, source
      - Backend-produced chunks (per shared_data/api_contracts): source_name,
        page, start_char/end_char, no document_title / chunk_index / word_count.

    Fields not present in the backend schema are derived with sensible
    fallbacks so retrieval never raises KeyError on mixed corpora.
    """
    if not query or not query.strip():
        raise ValueError('Query cannot be empty.')

    # Embed the query using the same model used for chunks
    query_vector = model.encode([query], convert_to_numpy=True)
    query_vector = np.array(query_vector, dtype='float32')

    safe_top_k = min(top_k, len(chunk_records))
    distances, indices = index.search(query_vector, safe_top_k)

    results = []
    for rank, chunk_idx in enumerate(indices[0]):
        chunk = chunk_records[int(chunk_idx)]
        distance_value = float(distances[0][rank])
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
            'rank': rank + 1,
            'document_id': chunk['document_id'],
            'document_title': document_title,
            'source': source,
            'chunk_id': chunk['chunk_id'],
            'chunk_index': chunk_index,
            'word_count': word_count,
            'distance': distance_value,
            'similarity': similarity_value,
            'citation': f"[{document_title} | {chunk['chunk_id']}]",
            'text': chunk['text'],
            'metadata': chunk.get('metadata', {})
        })

    return results


# ── Schema-fallback helpers ────────────────────────────────────────────────
# Mirrors bm25_retriever.py's helpers — kept local so this module stays
# self-contained (no cross-file import dependency for a small helper).
def _title_from_source(source: str) -> str:
    """Derive a human-readable title from a source filename when
    document_title is absent (backend-produced chunks).

    Strips the 8-char hex upload prefix and OCR markers first, matching
    the cleanup already applied to 'source'/'source_name' in the backend
    bridges (see _clean() in backend/app/retrieval/*.py).
    """
    import os, re
    cleaned = re.sub(r'^[0-9a-f]{8}_', '', source)
    cleaned = cleaned.replace('_ocr.txt', '.pdf').replace('_ocr', '')
    name = os.path.splitext(cleaned)[0]
    return name.replace('_', ' ').replace('-', ' ').title()


def _chunk_index_from_id(chunk_id: str) -> int:
    """Best-effort extraction of a numeric chunk index from chunk_id
    (e.g. 'doc-001-chunk-0004' -> 3) when chunk_index is absent."""
    import re
    match = re.search(r'(\d+)$', chunk_id or '')
    if match:
        return max(int(match.group(1)) - 1, 0)
    return 0