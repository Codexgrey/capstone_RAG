"""
src/evaluation/evaluate.py
Retrieval evaluation scripts.
Measures retrieval quality (Precision@K, Recall@K, MRR) and pipeline latency.

Two evaluation modes
--------------------
1. Runtime evaluation (always runs)
   Reports latency, similarity scores, rank distribution, and source coverage
   for every query — no ground truth required. This is what fires after every
   python -m src.main run.

2. Ground-truth evaluation (fires when the query is in TEST_QUERIES)
   Computes Precision@K, Recall@K, and MRR against known relevant chunk IDs.
   Used for formal benchmarking. Add queries to TEST_QUERIES as you build
   your test suite.

Test query bank
---------------
Each entry maps a natural language question to the chunk IDs that a correct
retrieval should surface. Chunk IDs follow the pattern '<doc_id>-chunk-N'.
Update these as your document set grows.

Usage
-----
    from src.evaluation.evaluate import run_evaluation
    run_evaluation(retrieved_results, query, latency_ms, top_k=3)
"""

import time
from typing import List, Dict, Any, Optional


# ---------------------------------------------------------------------------
# Test query bank — query -> list of relevant chunk IDs
# -These are updated to match actual documents and chunk IDs.
# ---------------------------------------------------------------------------
TEST_QUERIES: Dict[str, List[str]] = {
    'How does a vector retrieval RAG system answer a question?': [
        'doc-001-chunk-1',
        'doc-001-chunk-2',
        'doc-001-chunk-3',
    ],
    'What is the similarity score formula used in this system?': [
        'doc-001-chunk-5',
        'doc-001-chunk-3',
    ],
    'How is the FAISS index built and queried?': [
        'doc-001-chunk-3',
        'doc-001-chunk-2',
    ],
    'What metadata is stored with each chunk?': [
        'doc-001-chunk-4',
        'doc-001-chunk-2',
    ],
    'How does the prompt builder structure the generator input?': [
        'doc-001-chunk-6',
        'doc-001-chunk-5',
    ],
}


# ---------------------------------------------------------------------------
# core metric functions
# ---------------------------------------------------------------------------

def precision_at_k(retrieved_chunk_ids: List[str], relevant_chunk_ids: List[str], k: int) -> float:
    """
    Fraction of the top-k retrieved chunks that are relevant.

    Args:
        retrieved_chunk_ids: Ordered list of retrieved chunk IDs (best first).
        relevant_chunk_ids:  Chunk IDs considered relevant for this query.
        k:                   Cut-off rank.

    Returns:
        Precision@K in [0.0, 1.0].
    """
    top_k        = retrieved_chunk_ids[:k]
    relevant_set = set(relevant_chunk_ids)
    hits         = sum(1 for cid in top_k if cid in relevant_set)
    return hits / k if k > 0 else 0.0


def recall_at_k(retrieved_chunk_ids: List[str], relevant_chunk_ids: List[str], k: int) -> float:
    """
    Fraction of all relevant chunks that appear in the top-k results.

    Args:
        retrieved_chunk_ids: Ordered list of retrieved chunk IDs (best first).
        relevant_chunk_ids:  Chunk IDs considered relevant for this query.
        k:                   Cut-off rank.

    Returns:
        Recall@K in [0.0, 1.0].
    """
    top_k        = retrieved_chunk_ids[:k]
    relevant_set = set(relevant_chunk_ids)
    hits         = sum(1 for cid in top_k if cid in relevant_set)
    return hits / len(relevant_set) if relevant_set else 0.0


def mean_reciprocal_rank(retrieved_chunk_ids: List[str], relevant_chunk_ids: List[str]) -> float:
    """
    Reciprocal of the rank position of the first relevant chunk.

    Args:
        retrieved_chunk_ids: Ordered list of retrieved chunk IDs (best first).
        relevant_chunk_ids:  Chunk IDs considered relevant for this query.

    Returns:
        MRR in (0.0, 1.0], or 0.0 if no relevant chunk appears.
    """
    relevant_set = set(relevant_chunk_ids)
    for rank, cid in enumerate(retrieved_chunk_ids, start=1):
        if cid in relevant_set:
            return 1.0 / rank
    return 0.0


def evaluate_retrieval(
    results: List[Dict[str, Any]],
    relevant_chunk_ids: List[str],
    k: int = 3,
) -> Dict[str, float]:
    """
    Compute Precision@K, Recall@K, and MRR for a single query result set.

    Args:
        results:            Retriever output — list of result dicts ordered by rank.
        relevant_chunk_ids: Ground-truth relevant chunk IDs for this query.
        k:                  Cut-off rank for precision and recall.

    Returns:
        Dict with keys: precision_at_<k>, recall_at_<k>, mrr.
    """
    retrieved_ids = [r['chunk_id'] for r in results]
    return {
        f'precision_at_{k}': precision_at_k(retrieved_ids, relevant_chunk_ids, k),
        f'recall_at_{k}':    recall_at_k(retrieved_ids, relevant_chunk_ids, k),
        'mrr':               mean_reciprocal_rank(retrieved_ids, relevant_chunk_ids),
    }


# ---------------------------------------------------------------------------
# latency helper
# ---------------------------------------------------------------------------

class Timer:
    """Simple context-manager stopwatch that stores elapsed milliseconds."""

    def __init__(self) -> None:
        self.elapsed_ms: float = 0.0

    def __enter__(self) -> 'Timer':
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_: Any) -> None:
        self.elapsed_ms = (time.perf_counter() - self._start) * 1000


# ---------------------------------------------------------------------------
# runtime evaluation — always runs, no ground truth required
# ---------------------------------------------------------------------------

def print_runtime_report(
    query: str,
    results: List[Dict[str, Any]],
    latency_ms: float,
) -> None:
    """
    Print a runtime evaluation report for every query run.

    Reports latency, per-result similarity scores, and source document coverage.
    This fires on every run regardless of whether the query is in TEST_QUERIES.

    Args:
        query:      The natural language query that was run.
        results:    Retriever output list (ordered by rank).
        latency_ms: End-to-end retrieval latency in milliseconds.
    """
    sep     = '-' * 110
    top_k   = len(results)

    # similarity stats
    scores      = [r['similarity'] for r in results]
    avg_sim     = sum(scores) / len(scores) if scores else 0.0
    top_sim     = max(scores) if scores else 0.0
    bottom_sim  = min(scores) if scores else 0.0

    # source coverage — unique documents in results
    sources = list(dict.fromkeys(r['document_title'] for r in results))

    print(f'\nEVALUATION REPORT')
    print('=' * 110)
    print(
        f'Query  : {query}'
        f'Top-K  : {top_k}'
    )
    print(sep)

    # per-result breakdown
    print('Retrieved Chunks:')
    for r in results:
        bar = chr(9608) * int(r['similarity'] * 20)
        print(
            f"  Rank {r['rank']}  |  {r['chunk_id']:<30}  |  "
            f"Similarity: {r['similarity']:.4f}  {bar}"
        )
    print(sep)

    # similarity summary
    print('Similarity Summary:')
    print(
        f'  {"Top score:" :<20} {top_sim:.4f} \n'
        f'  {"Bottom score:" :<20} {bottom_sim:.4f} \n'
        f'  {"Average score:" :<20} {avg_sim:.4f} \n'
    )
    print(sep)

    # source coverage
    print(f'Source Coverage  : {len(sources)} document(s) represented in top-{top_k}')
    for s in sources:
        print(f'  - {s}')
    print(sep)

    # latency
    print(f'  {"Latency (ms)":<22} {latency_ms:.1f} ms')
    print(sep)


# ---------------------------------------------------------------------------
# ground-truth evaluation — fires when query is in TEST_QUERIES
# ---------------------------------------------------------------------------

def print_ground_truth_report(
    query: str,
    metrics: Dict[str, float],
    latency_ms: float,
    relevant_chunk_ids: List[str],
    retrieved_ids: List[str],
) -> None:
    """
    Print the formal ground-truth evaluation report (Precision, Recall, MRR).

    Args:
        query:              The evaluated query string.
        metrics:            Dict returned by evaluate_retrieval().
        latency_ms:         End-to-end retrieval latency in milliseconds.
        relevant_chunk_ids: Ground-truth chunk IDs.
        retrieved_ids:      Actual retrieved chunk IDs.
    """
    sep = '-' * 110
    print(f'\nGROUND-TRUTH EVALUATION')
    print('=' * 110)

    print(f'Query : {query}')
    print(sep)

    print(
        f'Ground-truth relevant chunks : {relevant_chunk_ids}'
        f'Retrieved chunk IDs          : {retrieved_ids}'
    )
    print(sep)

    for metric_name, score in metrics.items():
        label = metric_name.replace('_', ' ').upper()
        bar   = chr(9608) * int(score * 20)
        print(f'  {label:<22} {score:.4f}   {bar}')

    print(f'\n  {"Latency (ms)":<22} {latency_ms:.1f} ms')
    print(sep)


# ---------------------------------------------------------------------------
# top-level runner — called from main.py
# ---------------------------------------------------------------------------

def run_evaluation(
    retrieved_results: List[Dict[str, Any]],
    query: str,
    latency_ms: float,
    k: int = 3,
) -> Dict[str, Any]:
    """
    Run both evaluation modes for a query:
      1. Runtime report   — always prints (latency, similarity, source coverage).
      2. Ground-truth     — prints only if the query is in TEST_QUERIES.

    Args:
        retrieved_results: Retriever output list (ordered by rank).
        query:             The natural language query that was run.
        latency_ms:        End-to-end retrieval latency in milliseconds.
        k:                 Cut-off rank for precision and recall.

    Returns:
        Dict containing all computed metrics and latency_ms.
    """
    # 1. runtime report — always fires
    print_runtime_report(query, retrieved_results, latency_ms)

    metrics = {'latency_ms': latency_ms}

    # 2. ground-truth report — fires only when query is registered
    relevant_chunk_ids = TEST_QUERIES.get(query)
    if relevant_chunk_ids is not None:
        retrieved_ids  = [r['chunk_id'] for r in retrieved_results]
        gt_metrics     = evaluate_retrieval(retrieved_results, relevant_chunk_ids, k=k)
        gt_metrics['latency_ms'] = latency_ms
        metrics.update(gt_metrics)

        print_ground_truth_report(
            query=query,
            metrics={k_: v for k_, v in gt_metrics.items() if k_ != 'latency_ms'},
            latency_ms=latency_ms,
            relevant_chunk_ids=relevant_chunk_ids,
            retrieved_ids=retrieved_ids,
        )
    else:
        print(
            f'\n  [Ground-truth] Query not in TEST_QUERIES bank — '
            f'add it with known relevant chunk IDs to enable Precision/Recall/MRR scoring.'
        )

    return metrics