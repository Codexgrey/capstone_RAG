# ─── retrieval/hybrid_retriever.py ────────────────────────────────

def normalize_scores(results: list, score_key: str) -> list:
    """
    Normalizes scores to a 0-1 range using min-max normalization.
    """
    scores = [item[score_key] for item in results]
    min_score = min(scores) if scores else 0
    max_score = max(scores) if scores else 1

    # Avoid division by zero
    if max_score - min_score == 0:
        for item in results:
            item[f'{score_key}_normalized'] = 1.0
    else:
        for item in results:
            item[f'{score_key}_normalized'] = (
                (item[score_key] - min_score) / (max_score - min_score)
            )

    return results


def reciprocal_rank_fusion(
    bm25_results  : list,
    vector_results: list,
    k             : int = 60,
    top_k         : int = 3,
) -> list:
    """
    Combines BM25 and vector results using Reciprocal Rank Fusion (RRF).

    Formula :
        score_rrf = 1 / (k + rank_bm25) + 1 / (k + rank_vector)

    A chunk that ranks high in BOTH lists gets the highest final score.
    A chunk that only appears in one list still gets a partial score.

    Args :
        bm25_results   : ranked list from keyword retrieval
        vector_results : ranked list from vector retrieval
        k              : smoothing constant (default 60)
        top_k          : number of final results to return

    Returns :
        list of top-K chunks sorted by hybrid RRF score
    """

    # ─── Step 1 — Build rank maps ───────────────────────────────────
    # { chunk_id → rank } for each retrieval method
    bm25_ranks   = {
        item['chunk_id']: item['rank']
        for item in bm25_results
    }
    vector_ranks = {
        item['chunk_id']: item['rank']
        for item in vector_results
    }

    # ─── Step 2 — Union of all chunks (merge + deduplicate) ─────────
    all_chunks = {}

    for item in bm25_results:
        cid = item['chunk_id']
        if cid not in all_chunks:
            all_chunks[cid] = item.copy()
            all_chunks[cid]['bm25_score']   = item.get('bm25_score', 0.0)
            all_chunks[cid]['similarity']   = None
            all_chunks[cid]['rrf_score']    = 0.0

    for item in vector_results:
        cid = item['chunk_id']
        if cid not in all_chunks:
            all_chunks[cid] = item.copy()
            all_chunks[cid]['bm25_score']   = None
            all_chunks[cid]['similarity']   = item.get('similarity', 0.0)
            all_chunks[cid]['rrf_score']    = 0.0
        else:
            all_chunks[cid]['similarity'] = item.get('similarity', 0.0)

    # ─── Step 3 — Compute RRF score for each chunk ──────────────────
    for cid, chunk in all_chunks.items():
        rrf = 0.0

        # BM25 contribution
        if cid in bm25_ranks:
            rrf += 1 / (k + bm25_ranks[cid])

        # Vector contribution
        if cid in vector_ranks:
            rrf += 1 / (k + vector_ranks[cid])

        chunk['rrf_score'] = rrf

    # ─── Step 4 — Sort by RRF score descending ──────────────────────
    sorted_chunks = sorted(
        all_chunks.values(),
        key     = lambda x: x['rrf_score'],
        reverse = True
    )

    # ─── Step 5 — Top-K selection + re-rank ─────────────────────────
    final_results = []
    for rank, chunk in enumerate(sorted_chunks[:top_k]):
        chunk['rank']      = rank + 1
        chunk['retrieval'] = _get_retrieval_source(
            chunk['chunk_id'],
            bm25_ranks,
            vector_ranks
        )
        final_results.append(chunk)

    return final_results


def _get_retrieval_source(chunk_id, bm25_ranks, vector_ranks) -> str:
    """
    Indicates which retrieval method found this chunk.
    """
    in_bm25   = chunk_id in bm25_ranks
    in_vector = chunk_id in vector_ranks

    if in_bm25 and in_vector:
        return 'BOTH (keyword + vector)'
    elif in_bm25:
        return 'keyword only'
    else:
        return 'vector only'