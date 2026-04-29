"""
evaluation
=======================
Retrieval evaluation scripts for the keyword (BM25) retrieval module.

Metrics implemented
-------------------
- Precision@K   : fraction of top-K results that are relevant
- Recall@K      : fraction of relevant docs found in top-K
- MRR           : Mean Reciprocal Rank
- Average NDCG  : Normalised Discounted Cumulative Gain
"""

import math


# ---------------------------------------------------------------------------
# Core metrics
# ---------------------------------------------------------------------------

def precision_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Precision@K — what fraction of the top-K results are relevant?"""
    top_k = retrieved_ids[:k]
    hits  = sum(1 for cid in top_k if cid in relevant_ids)
    return hits / k if k else 0.0


def recall_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Recall@K — what fraction of relevant docs appear in the top-K?"""
    if not relevant_ids:
        return 0.0
    top_k = retrieved_ids[:k]
    hits  = sum(1 for cid in top_k if cid in relevant_ids)
    return hits / len(relevant_ids)


def reciprocal_rank(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    """Reciprocal rank of the first relevant result."""
    for rank, cid in enumerate(retrieved_ids, start=1):
        if cid in relevant_ids:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """NDCG@K (binary relevance)."""
    dcg  = sum(
        1.0 / math.log2(rank + 1)
        for rank, cid in enumerate(retrieved_ids[:k], start=1)
        if cid in relevant_ids
    )
    idcg = sum(
        1.0 / math.log2(rank + 1)
        for rank in range(1, min(len(relevant_ids), k) + 1)
    )
    return dcg / idcg if idcg else 0.0


# ---------------------------------------------------------------------------
# Batch evaluation
# ---------------------------------------------------------------------------

def evaluate(
    results:      list[dict],
    relevant_ids: set[str],
    k:            int = 5,
) -> dict:
    """
    Compute all metrics for a single query result set.

    Parameters
    ----------
    results      : list of result dicts from ``retriever.retrieve``
    relevant_ids : set of chunk_ids considered relevant for this query
    k            : cut-off rank

    Returns
    -------
    dict with keys: precision_at_k, recall_at_k, mrr, ndcg_at_k
    """
    retrieved_ids = [r["chunk_id"] for r in results]
    return {
        f"precision@{k}": precision_at_k(retrieved_ids, relevant_ids, k),
        f"recall@{k}":    recall_at_k(retrieved_ids, relevant_ids, k),
        "mrr":            reciprocal_rank(retrieved_ids, relevant_ids),
        f"ndcg@{k}":      ndcg_at_k(retrieved_ids, relevant_ids, k),
    }


def evaluate_batch(
    query_results: list[tuple[list[dict], set[str]]],
    k:             int = 5,
) -> dict:
    """
    Average evaluation metrics over multiple queries.

    Parameters
    ----------
    query_results : list of (results, relevant_ids) tuples
    k             : cut-off rank

    Returns
    -------
    dict with averaged metrics
    """
    all_metrics: dict[str, list[float]] = {}
    for results, relevant_ids in query_results:
        m = evaluate(results, relevant_ids, k)
        for key, val in m.items():
            all_metrics.setdefault(key, []).append(val)

    return {key: sum(vals) / len(vals) for key, vals in all_metrics.items()}


# ---------------------------------------------------------------------------
# CLI helper
# ---------------------------------------------------------------------------

def print_metrics(metrics: dict) -> None:
    print("\n--- Retrieval Evaluation Metrics ---")
    for key, val in metrics.items():
        print(f"  {key:<20} {val:.4f}")
    print()
