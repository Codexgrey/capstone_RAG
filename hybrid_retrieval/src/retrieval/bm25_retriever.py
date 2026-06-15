from groq import Groq
try:
    # Backend bridge context: hybrid_retrieval/src is on sys.path,
    # so `preprocessing` is importable directly as a top-level package.
    from preprocessing.preprocess import tokenize_chunk
except ImportError:
    # Standalone context: `python -m src.main` from hybrid_retrieval/ root,
    # where `src` itself is the top-level package.
    from src.preprocessing.preprocess import tokenize_chunk

# Query Normaliser 
def normalise_query(raw_query: str, groq_client: Groq, model_name: str) -> str:
    """
    Uses a small LLM to extract the core search keywords from a query.
    Removes filler words, articles, and conversational phrases.
    Returns a cleaned keyword string suitable for BM25 lookup.
    """
    system_prompt = (
        'You are a query preprocessing assistant for a keyword-based search system. '
        'Given a user query, extract only the most important, content-bearing keywords. '
        'Remove filler words, articles, and conversational phrases. '
        'Return ONLY a space-separated list of keywords — no punctuation, no explanation.'
    )

    response = groq_client.chat.completions.create(
        model    = model_name,
        messages = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user',   'content': f'Query: {raw_query}'},
        ],
        max_tokens  = 100,
        temperature = 0.0,
    )

    return response.choices[0].message.content.strip()


# BM25 Retriever
def retrieve_bm25(
    query         : str,
    bm25,
    chunk_records : list,
    inverted_index: dict,
    nltk_lang     : str = 'english',
    top_k         : int = 5,
) -> list:
    """
    Tokenises the query with the same pipeline used on chunks,
    scores all chunks using BM25, and returns the top-K results.

    chunk_records may contain two different schemas:
      - Research-pipeline chunks (from this module's own ingest):
        document_title, chunk_index, word_count, source
      - Backend-produced chunks (from app/ingestion/chunker.py, per
        shared_data/api_contracts): source_name, page, start_char/end_char,
        no document_title / chunk_index / word_count.

    Fields not present in the backend schema are derived with sensible
    fallbacks so retrieval never raises KeyError on mixed corpora.
    """
    if not query or not query.strip():
        raise ValueError('Query cannot be empty.')

    # Tokenise query same way as chunks
    query_tokens = tokenize_chunk(query, nltk_lang)

    if not query_tokens:
        return []

    # BM25 scoring
    scores  = bm25.get_scores(query_tokens)
    safe_k  = min(top_k, len(chunk_records))
    top_idxs = sorted(
        range(len(scores)),
        key     = lambda i: scores[i],
        reverse = True
    )[:safe_k]

    results = []
    for rank, idx in enumerate(top_idxs):
        chunk = chunk_records[idx]
        score = float(scores[idx])

        source = chunk.get('source') or chunk.get('source_name') or 'Unknown'
        document_title = chunk.get('document_title') or _title_from_source(source)
        chunk_index = chunk.get('chunk_index')
        if chunk_index is None:
            chunk_index = _chunk_index_from_id(chunk.get('chunk_id', ''))
        word_count = chunk.get('word_count')
        if word_count is None:
            word_count = len(chunk.get('text', '').split())

        # Collect which query terms actually hit this chunk
        matched_terms = [
            t for t in query_tokens
            if t in inverted_index
            and any(p['chunk_idx'] == idx for p in inverted_index[t]['postings'])
        ]

        results.append({
            'rank'          : rank + 1,
            'document_id'   : chunk['document_id'],
            'document_title': document_title,
            'source'        : source,
            'chunk_id'      : chunk['chunk_id'],
            'chunk_index'   : chunk_index,
            'word_count'    : word_count,
            'bm25_score'    : score,
            'matched_terms' : matched_terms,
            'citation'      : f"[{document_title} | {chunk['chunk_id']}]",
            'text'          : chunk['text'],
        })

    return results


# ── Schema-fallback helpers ────────────────────────────────────────────────
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
    (e.g. 'doc-001-chunk-0004' -> 3) when chunk_index is absent."""
    import re
    match = re.search(r'(\d+)$', chunk_id or '')
    if match:
        return max(int(match.group(1)) - 1, 0)
    return 0