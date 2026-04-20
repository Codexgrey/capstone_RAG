"""
src/evaluation/evaluate.py
Retrieval evaluation scripts.
Measures retrieval quality (Precision@K, Recall@K, MRR) and pipeline latency
against a set of test queries with known relevant chunk IDs.

Test query bank
---------------
Each entry maps a natural language question to the chunk IDs from sample.txt
that a correct retrieval should surface. Chunk IDs follow the pattern
'doc-001-chunk-N' where N is 1-indexed (set by the chunker).

    Chunk 1  — RAG overview: two-stage design, LLM constrained by retrieval
    Chunk 2  — Retrieval stage: sentence-transformer encoding, vector space
    Chunk 3  — FAISS index: nearest-neighbour search, top-k
    Chunk 4  — Chunk metadata: document_id, title, source, citations
    Chunk 5  — Similarity score: L2 distance -> similarity formula
    Chunk 6  — Prompt builder + generator (Groq, llama-3.1-8b-instant)

Usage
-----
    from src.evaluation.evaluate import run_evaluation
    run_evaluation(retrieved_results, query, latency_ms, top_k=3)
"""

import time
from typing import List, Dict, Any, Optional


# ---------------------------------------------------------------------------
# Test query bank — query -> list of relevant chunk IDs
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
# Core metric functions
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
    top_k = retrieved_chunk_ids[:k]
    relevant_set = set(relevant_chunk_ids)
    hits = sum(1 for cid in top_k if cid in relevant_set)
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
    top_k = retrieved_chunk_ids[:k]
    relevant_set = set(relevant_chunk_ids)
    hits = sum(1 for cid in top_k if cid in relevant_set)
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
    Compute all retrieval metrics for a single query result set.

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
# report printer
# ---------------------------------------------------------------------------

def print_evaluation_report(
    query: str,
    metrics: Dict[str, float],
    latency_ms: float,
    relevant_chunk_ids: Optional[List[str]] = None,
    retrieved_ids: Optional[List[str]] = None,
) -> None:
    """
    Print a formatted evaluation report to stdout.

    Args:
        query:              The evaluated query string.
        metrics:            Dict returned by evaluate_retrieval().
        latency_ms:         End-to-end retrieval latency in milliseconds.
        relevant_chunk_ids: Ground-truth chunk IDs (printed for reference).
        retrieved_ids:      Actual retrieved chunk IDs (printed for reference).
    """
    sep = '-' * 110
    print(f'\nEVALUATION REPORT')
    print('=' * 110)
    print(f'Query : {query}')
    print(sep)

    if relevant_chunk_ids is not None:
        print(f'Ground-truth relevant chunks : {relevant_chunk_ids}')
    if retrieved_ids is not None:
        print(f'Retrieved chunk IDs          : {retrieved_ids}')
    print(sep)

    for metric_name, score in metrics.items():
        label = metric_name.replace('_', ' ').upper()
        bar   = chr(9608) * int(score * 20)
        print(f'  {label:<22} {score:.4f}   {bar}')

    print(f'\n  {"LATENCY (ms)":<22} {latency_ms:.1f} ms')
    print(sep)


# ---------------------------------------------------------------------------
# Top-level runner — called from main.py
# ---------------------------------------------------------------------------

def run_evaluation(
    retrieved_results: List[Dict[str, Any]],
    query: str,
    latency_ms: float,
    k: int = 3,
) -> Dict[str, Any]:
    """
    Look up ground-truth for the query, compute all metrics, and print
    the evaluation report.

    If the query is not in the TEST_QUERIES bank the function still reports
    latency and notes that no ground-truth is available — it does not crash.

    Args:
        retrieved_results: Retriever output list (ordered by rank).
        query:             The natural language query that was run.
        latency_ms:        End-to-end retrieval latency in milliseconds.
        k:                 Cut-off rank for precision and recall.

    Returns:
        Dict containing metrics and latency_ms, suitable for logging.
    """
    relevant_chunk_ids = TEST_QUERIES.get(query)
    retrieved_ids = [r['chunk_id'] for r in retrieved_results]

    if relevant_chunk_ids is None:
        print(f'\nEVALUATION NOTE: query not found in TEST_QUERIES bank.')
        print(f'  Latency: {latency_ms:.1f} ms  |  Retrieved: {retrieved_ids}')
        return {'latency_ms': latency_ms}

    metrics = evaluate_retrieval(retrieved_results, relevant_chunk_ids, k=k)
    metrics['latency_ms'] = latency_ms

    print_evaluation_report(
        query=query,
        metrics={k_: v for k_, v in metrics.items() if k_ != 'latency_ms'},
        latency_ms=latency_ms,
        relevant_chunk_ids=relevant_chunk_ids,
        retrieved_ids=retrieved_ids,
    )

    return metrics
