from groq import Groq
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

        # Collect which query terms actually hit this chunk
        matched_terms = [
            t for t in query_tokens
            if t in inverted_index
            and any(p['chunk_idx'] == idx for p in inverted_index[t]['postings'])
        ]

        results.append({
            'rank'          : rank + 1,
            'document_id'   : chunk['document_id'],
            'document_title': chunk['document_title'],
            'source'        : chunk['source'],
            'chunk_id'      : chunk['chunk_id'],
            'chunk_index'   : chunk['chunk_index'],
            'word_count'    : chunk['word_count'],
            'bm25_score'    : score,
            'matched_terms' : matched_terms,
            'citation'      : f"[{chunk['document_title']} | {chunk['chunk_id']}]",
            'text'          : chunk['text'],
        })

    return results