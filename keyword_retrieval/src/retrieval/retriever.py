"""
retrieval/retriever.py
=======================
Keyword search and BM25 ranking — returns the top-K most relevant chunks.

This module is called by main.py at Step 8.

What this module does
---------------------
Given a normalised query (from Step 7) and the BM25 model + inverted
index (from Step 6), it:

  1. Tokenises the query using the same pipeline as the chunks
     (lowercase → remove stopwords → stem).

  2. Scores every chunk using BM25.get_scores(query_tokens).
     Each chunk gets a float score — higher means more relevant.

  3. Sorts chunks by score (highest first) and returns the top-K.

  4. For each result, reports which query tokens actually matched
     in that chunk (matched_terms) — useful for transparency and
     debugging.

Result structure (each result is a dict)
-----------------------------------------
  {
    "rank":           int    position in the ranked list (1 = best)
    "document_id":    str    e.g. "doc-001"
    "document_title": str    human-readable document name
    "source":         str    original file path or URL
    "chunk_id":       str    e.g. "doc-001-chunk-3"
    "chunk_index":    int    0-based position in the chunk list
    "word_count":     int    number of words in this chunk
    "bm25_score":     float  relevance score from BM25
    "matched_terms":  list   query tokens that hit this chunk
    "citation":       str    e.g. "[My Doc | doc-001-chunk-3]"
    "text":           str    the actual chunk text
  }

One public function
-------------------
  retrieve(query, bm25, chunk_records, inverted_index, nltk_lang, top_k)
      Runs the full search and returns a ranked list of result dicts.
"""

from preprocessing.preprocess import tokenize_chunk


# =============================================================================
# STEP 8 — RETRIEVE TOP-K CHUNKS
# =============================================================================

def retrieve(
    query:          str,
    bm25,
    chunk_records:  list[dict],
    inverted_index: dict,
    nltk_lang:      str = "english",
    top_k:          int = 5,
) -> list[dict]:
    """
    Search the index and return the top-K most relevant chunks.

    How it works:
        1. Tokenise the query the same way the chunks were tokenised
           (ensures query tokens match index tokens exactly).
        2. Call bm25.get_scores(query_tokens) to score every chunk.
        3. Sort all chunks by score descending.
        4. Take the top-K results.
        5. For each result, check the inverted index to find which
           query tokens actually appear in that chunk (matched_terms).
        6. Return a list of result dicts with all metadata attached.

    Parameters
    ----------
    query          : str
        The normalised query string from Step 7.
        Can be raw keywords or a full sentence — it will be tokenised.

    bm25           : BM25Okapi
        The fitted BM25 model from Step 6 (bm25_store.py).

    chunk_records  : list[dict]
        The chunk metadata dicts from Step 4 (chunker.py).
        Must be in the same order used to build the BM25 model.

    inverted_index : dict
        The inverted index from Step 6 (indexer.py).
        Used to look up which query tokens appear in each chunk.

    nltk_lang      : str
        NLTK stopword corpus name from Step 3 (e.g. 'english').
        Must match the language used when tokenising the chunks.

    top_k          : int
        Number of chunks to return. Recommended: 3–7.

    Returns
    -------
    list[dict]
        Ranked list of result dicts. Best match first (rank 1).
        Empty list if no tokens remain after tokenisation.

    Raises
    ------
    ValueError
        If query is empty or whitespace only.

    Example
    -------
        >>> results = retrieve("how does BM25 work", bm25, chunks, index)
        >>> results[0]["rank"]
        1
        >>> results[0]["bm25_score"]
        0.7431
    """
    # --- Validate query ---
    if not query or not query.strip():
        raise ValueError("Query cannot be empty.")

    # --- Step 1: tokenise the query ---
    # Uses the same pipeline as the chunks so tokens match the index.
    query_tokens = tokenize_chunk(query, nltk_lang)

    # If all words were stopwords or non-alpha, nothing to search
    if not query_tokens:
        print("  [Retriever] Warning: no searchable tokens in query after cleaning.")
        return []

    # --- Step 2: score every chunk with BM25 ---
    scores = bm25.get_scores(query_tokens)
    # scores is a numpy array with one float per chunk,
    # in the same order as chunk_records.

    # --- Step 3: sort by score and take top-K ---
    # Clamp top_k to the number of chunks we actually have
    safe_k   = min(top_k, len(chunk_records))

    # Get indices of the top-K scores (highest first)
    top_idxs = sorted(
        range(len(scores)),
        key     = lambda i: scores[i],
        reverse = True,
    )[:safe_k]

    # --- Step 4: build result dicts ---
    results = []

    for rank, chunk_idx in enumerate(top_idxs):

        chunk = chunk_records[chunk_idx]
        score = float(scores[chunk_idx])

        # --- Step 5: find which query tokens matched this chunk ---
        # A token "matches" if it is in the inverted index AND
        # has a posting for this specific chunk.
        matched_terms = [
            token for token in query_tokens
            if token in inverted_index
            and any(
                posting["chunk_idx"] == chunk_idx
                for posting in inverted_index[token]["postings"]
            )
        ]

        # Build the result dict
        result = {
            "rank":           rank + 1,
            "document_id":    chunk["document_id"],
            "document_title": chunk["document_title"],
            "source":         chunk["source"],
            "chunk_id":       chunk["chunk_id"],
            "chunk_index":    chunk["chunk_index"],
            "word_count":     chunk["word_count"],
            "bm25_score":     score,
            "matched_terms":  matched_terms,
            "citation":       f"[{chunk['document_title']} | {chunk['chunk_id']}]",
            "text":           chunk["text"],
        }

        results.append(result)

    return results
