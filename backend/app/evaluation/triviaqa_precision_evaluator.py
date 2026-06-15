"""
backend/app/evaluation/triviaqa_precision_evaluator.py
=======================================================
Precision/Recall@k benchmark for the Vector, Keyword, and Hybrid retrieval
modules, using TriviaQA's own per-question evidence as the gold corpus.

How this differs from triviaqa_evaluator.py
--------------------------------------------
triviaqa_evaluator.py scores generated answers (EM/F1) against the team's
own 9-document corpus, using a static 5,000-question bank that carries no
evidence text. That bank cannot support a Precision/Recall metric, because
there is no "gold document" to compare retrieved chunks against.

This module addresses that by streaming TriviaQA (rc, validation) directly
from HuggingFace. Each streamed record carries its own evidence text
(Wikipedia entity pages and/or web search-result contexts). For a given
question, every chunk built from that question's evidence is, by
construction, a gold-relevant chunk — TriviaQA guarantees the answer is
supported somewhere in that evidence.

Per-question pipeline (isolated evaluation)
--------------------------------------------
For each question:
  1. Build chunks from this question's own evidence (entity_pages +
     search_results), using the same fixed 200-word / 40-word-overlap
     chunker as production (backend/app/ingestion/chunker.py).
  2. ingest(chunks, document_id) via the real adapter for the chosen
     retrieval method.
  3. retrieve(query, top_k) via the real adapter, timed.
  4. Score the SAME top-k result set two ways:
       - Answer-in-Context@5 / Answer-in-Top-1 / MRR
         (does any returned chunk contain a correct answer alias?)
       - Precision@k / Recall@k
         (how many returned chunks belong to this question's own
         evidence, i.e. are gold-relevant, vs. total returned / total gold)
  5. delete(document_id) from the index(es) — nothing carries over to
     the next question.

This mirrors the isolated-evaluation design used for the existing BM25
groundedness benchmark, generalised to all three adapters and extended
with the Precision/Recall scoring described above.

Gold-relevance scoping
-----------------------
"Gold document" for a question = search_results.search_context (paired
with search_results.title), the evidence source verified against a
working reference implementation of this benchmark. entity_pages.wiki_context
is used as a fallback only when search_results is empty for a record, so
no question is skipped purely because it lacks search-result evidence.
No domain-based restriction is applied beyond this fallback ordering.

Adapter requirements
----------------------
Vector and Keyword adapters expose ingest()/retrieve()/delete(). Hybrid
has no index of its own — it fuses the FAISS index (vector) and BM25
pickle (keyword) on disk — so cleanup for Hybrid runs is performed by
calling delete() on both the vector and keyword adapters; Hybrid's own
retrieve() then reflects the cleared state on its next call.

Resumability
-------------
Streaming datasets do not support random access — resuming at question N
means iterating past the first N records again. Progress is checkpointed
to disk after every question (resume index + running totals), matching
the batch-by-batch session pattern used for the existing 5,000-question
BM25 groundedness run.

Public API
----------
    run_precision_batch(retrieval_method, top_k, batch_size,
                         checkpoint_path) -> dict
    get_checkpoint_status(checkpoint_path) -> dict
    reset_checkpoint(checkpoint_path) -> None
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Reuse the official TriviaQA answer-normalisation logic so alias matching
# behaves identically to the groundedness benchmark.
from app.evaluation.triviaqa_evaluator import _normalize_answer

# Real adapters — same bridges used by the live chat pipeline.
from app.retrieval import vector_adapter, keyword_adapter, hybrid_adapter

# Production chunker — same fixed 200-word / 40-word-overlap convention
# used for the team's own document corpus.
from app.ingestion.chunker import chunk_text, DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP


# ---------------------------------------------------------------------------
# Adapter dispatch
# ---------------------------------------------------------------------------

_ADAPTERS = {
    "vector":  vector_adapter,
    "keyword": keyword_adapter,
    "hybrid":  hybrid_adapter,
}

_VALID_METHODS = tuple(_ADAPTERS.keys())


def _cleanup(document_id: str, adapter=None) -> None:
    """
    Remove a question's evidence from every index that might hold it.

    Hybrid has no index of its own — it reads the FAISS index and BM25
    pickle written by the vector and keyword adapters. Clearing both
    unconditionally keeps all three retrieval_method runs isolated from
    each other regardless of which method is currently being benchmarked.

    When running the Hybrid track, the hybrid bridge's in-memory
    FAISS/BM25 caches must also be dropped after delete() rewrites those
    files on disk, so the next question's retrieve() reloads the cleared
    indexes rather than serving stale cached state.
    """
    vector_adapter.delete(document_id=document_id)
    keyword_adapter.delete(document_id=document_id)
    if adapter is hybrid_adapter:
        hybrid_adapter._invalidate_cache()


# ---------------------------------------------------------------------------
# Answer matching (mirrors the existing groundedness scoring logic)
# ---------------------------------------------------------------------------

def _answer_in_text(text: str, aliases: List[str]) -> bool:
    """True if any normalised alias appears as a substring of the
    normalised chunk text."""
    if not text or not aliases:
        return False
    norm_text = _normalize_answer(text)
    return any(_normalize_answer(alias) in norm_text for alias in aliases if alias)


def _get_aliases(record: Dict[str, Any]) -> List[str]:
    """
    Collect every valid answer form for a record: the alias list plus the
    primary value. Field names (aliases, value) match the live trivia_qa
    "rc" schema as consumed directly from the HF stream.
    """
    answer = record.get("answer", {}) or {}
    aliases = list(answer.get("aliases") or [])
    value = answer.get("value")
    if value and value not in aliases:
        aliases.append(value)
    return [a for a in aliases if a and a.strip()]


# ---------------------------------------------------------------------------
# Evidence -> chunks
# ---------------------------------------------------------------------------

def build_chunks_from_evidence(record: Dict[str, Any], document_id: str,
                                chunk_size: int = DEFAULT_CHUNK_SIZE,
                                chunk_overlap: int = DEFAULT_CHUNK_OVERLAP
                                ) -> List[Dict[str, Any]]:
    """
    Build chunks from a TriviaQA "rc" record's own evidence.

    Primary evidence source is search_results.search_context (paired with
    search_results.title), matching the field names actually present on
    live trivia_qa "rc" records. If search_results is empty for a record,
    entity_pages.wiki_context is used as a fallback so questions that only
    carry Wikipedia evidence are not skipped outright.

    Every resulting chunk is gold-relevant for this question by
    construction, since it comes from this question's own evidence and
    nothing else.

    Returns an empty list if the record has no usable evidence text at
    all — callers should skip such records (cannot be scored for
    Precision/Recall).
    """
    texts: List[str] = []

    search_results = record.get("search_results") or {}
    search_contexts = search_results.get("search_context") or []
    search_titles   = search_results.get("title") or []

    for i, ctx in enumerate(search_contexts):
        if ctx and ctx.strip():
            title = search_titles[i] if i < len(search_titles) else ""
            texts.append(f"{title}\n\n{ctx}" if title else ctx)

    if not texts:
        entity_pages = record.get("entity_pages") or {}
        wiki_contexts = entity_pages.get("wiki_context") or []
        wiki_titles   = entity_pages.get("title") or []
        for i, ctx in enumerate(wiki_contexts):
            if ctx and ctx.strip():
                title = wiki_titles[i] if i < len(wiki_titles) else ""
                texts.append(f"{title}\n\n{ctx}" if title else ctx)

    if not texts:
        return []

    combined = "\n\n[PAGE 1]\n".join(texts) if len(texts) > 1 else texts[0]
    if not combined.startswith("[PAGE"):
        combined = "[PAGE 1]\n" + combined

    try:
        return chunk_text(
            text=combined,
            document_id=document_id,
            source_name=f"triviaqa-evidence-{record.get('question_id', 'unknown')}",
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
    except ValueError:
        # Empty/whitespace-only text after stripping — no usable evidence.
        return []


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

_DEFAULT_CHECKPOINT = Path(__file__).parent / "triviaqa_precision_checkpoint.json"


def _default_totals() -> Dict[str, Any]:
    return {
        "n_scored":            0,
        "n_skipped_no_evidence": 0,
        "sum_answer_at_k":     0.0,
        "sum_answer_top1":     0.0,
        "sum_reciprocal_rank": 0.0,
        "sum_precision_at_k":  0.0,
        "sum_recall_at_k":     0.0,
        "sum_latency_ms":      0.0,
        "checkpoints":         [],   # progress snapshots, written every 50 questions
    }


def _checkpoint_path(method: str, checkpoint_path: Optional[str]) -> Path:
    if checkpoint_path:
        return Path(checkpoint_path)
    return _DEFAULT_CHECKPOINT.with_name(f"triviaqa_precision_checkpoint_{method}.json")


def _load_checkpoint(path: Path) -> Dict[str, Any]:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"resume_index": 0, "totals": _default_totals()}


def _save_checkpoint(path: Path, state: Dict[str, Any]) -> None:
    path.write_text(json.dumps(state, indent=2), encoding="utf-8")


def get_checkpoint_status(retrieval_method: str,
                           checkpoint_path: Optional[str] = None) -> Dict[str, Any]:
    """Return the current resume position and running metrics without
    running any questions — used by the frontend to show "N done, click
    Run to continue"."""
    path  = _checkpoint_path(retrieval_method, checkpoint_path)
    state = _load_checkpoint(path)
    return _summarize(state, retrieval_method, target=None)


def reset_checkpoint(retrieval_method: str,
                      checkpoint_path: Optional[str] = None) -> None:
    """Delete the checkpoint file for a method, starting that benchmark
    over from question 0."""
    path = _checkpoint_path(retrieval_method, checkpoint_path)
    if path.exists():
        path.unlink()


# ---------------------------------------------------------------------------
# Per-question pipeline
# ---------------------------------------------------------------------------

def _run_one_question(record: Dict[str, Any], adapter, top_k: int,
                       chunk_size: int, chunk_overlap: int
                       ) -> Optional[Dict[str, Any]]:
    """
    Run ingest -> retrieve -> score -> delete for a single TriviaQA record.

    Returns None if the record has no usable evidence (skipped, does not
    count toward n_scored).
    """
    question_id = record.get("question_id", "unknown")
    document_id = f"triviaqa-precision-{question_id}"

    chunks = build_chunks_from_evidence(record, document_id, chunk_size, chunk_overlap)
    if not chunks:
        return None

    gold_chunk_ids = {c["chunk_id"] for c in chunks}

    try:
        # Hybrid has no index of its own — its ingest() is a no-op readiness
        # check (it just confirms faiss_index.bin / keyword_bm25.pkl exist).
        # To actually get this question's evidence into the indexes Hybrid
        # fuses at query time, ingest into vector AND keyword directly.
        # retrieve() still goes through the requested adapter (hybrid ->
        # RRF fusion over both).
        if adapter is hybrid_adapter:
            vector_adapter.ingest(chunks=chunks, document_id=document_id)
            keyword_adapter.ingest(chunks=chunks, document_id=document_id)
            # Force the hybrid bridge to reload its module on the next
            # retrieve(), so its in-memory FAISS/BM25 caches reflect the
            # indexes vector/keyword adapters just rewrote on disk.
            hybrid_adapter._invalidate_cache()
        else:
            adapter.ingest(chunks=chunks, document_id=document_id)

        start = time.perf_counter()
        result = adapter.retrieve(query=record["question"], top_k=top_k)
        latency_ms = (time.perf_counter() - start) * 1000

        # retrieve() returns either {"results": [...]} (vector/keyword/hybrid
        # bridge dict form) or a bare list, depending on adapter — normalise.
        results = result.get("results", result) if isinstance(result, dict) else result
        results = results or []

        aliases = _get_aliases(record)

        answer_hit  = any(_answer_in_text(c.get("text", ""), aliases) for c in results)
        answer_top1 = bool(results) and _answer_in_text(results[0].get("text", ""), aliases)

        reciprocal_rank = 0.0
        for rank, c in enumerate(results, start=1):
            if _answer_in_text(c.get("text", ""), aliases):
                reciprocal_rank = 1.0 / rank
                break

        returned_ids = [c.get("chunk_id") for c in results]
        gold_returned = sum(1 for cid in returned_ids if cid in gold_chunk_ids)

        precision_at_k = (gold_returned / top_k) if top_k else 0.0
        recall_at_k    = (gold_returned / len(gold_chunk_ids)) if gold_chunk_ids else 0.0

        return {
            "question_id":     question_id,
            "answer_at_k":     1.0 if answer_hit else 0.0,
            "answer_top1":     1.0 if answer_top1 else 0.0,
            "reciprocal_rank": reciprocal_rank,
            "precision_at_k":  precision_at_k,
            "recall_at_k":     recall_at_k,
            "latency_ms":      latency_ms,
            "n_gold_chunks":   len(gold_chunk_ids),
            "n_returned":      len(results),
        }

    finally:
        # Always attempt cleanup, even if ingest/retrieve raised, so a
        # failed question doesn't leave evidence in the index for the
        # next one.
        _cleanup(document_id, adapter=adapter)


# ---------------------------------------------------------------------------
# Streaming + batch runner
# ---------------------------------------------------------------------------

def _stream_trivia_qa():
    """Lazy import — datasets/HF access is only needed when actually
    running a batch, not at module import time."""
    from datasets import load_dataset
    return load_dataset("trivia_qa", "rc", split="validation", streaming=True)


def run_precision_batch(
    retrieval_method: str,
    top_k:            int            = 5,
    batch_size:       int            = 200,
    chunk_size:       int            = DEFAULT_CHUNK_SIZE,
    chunk_overlap:    int            = DEFAULT_CHUNK_OVERLAP,
    checkpoint_path:  Optional[str]  = None,
) -> Dict[str, Any]:
    """
    Run the next `batch_size` questions of the Precision/Recall benchmark
    for `retrieval_method`, resuming from the saved checkpoint.

    Args:
        retrieval_method : "vector", "keyword", or "hybrid".
        top_k            : Chunks to retrieve per question.
        batch_size       : Number of TriviaQA records to process this call
                            (records with no evidence are skipped but still
                            advance the resume position).
        chunk_size       : Words per chunk (default matches production: 200).
        chunk_overlap    : Overlap words between chunks (default: 40).
        checkpoint_path  : Override path for the checkpoint file. Defaults
                            to a per-method file alongside this module.

    Returns:
        Summary dict (see _summarize) with cumulative metrics across all
        questions processed so far, plus this batch's question count.
    """
    if retrieval_method not in _VALID_METHODS:
        raise ValueError(f"retrieval_method must be one of {_VALID_METHODS}, "
                         f"got {retrieval_method!r}")

    adapter = _ADAPTERS[retrieval_method]
    path    = _checkpoint_path(retrieval_method, checkpoint_path)
    state   = _load_checkpoint(path)
    totals  = state["totals"]

    resume_index   = state["resume_index"]
    target_index   = resume_index + batch_size
    this_batch_n   = 0

    stream = _stream_trivia_qa()

    for i, record in enumerate(stream):
        if i < resume_index:
            continue
        if i >= target_index:
            break

        outcome = _run_one_question(record, adapter, top_k, chunk_size, chunk_overlap)
        this_batch_n += 1

        if outcome is None:
            totals["n_skipped_no_evidence"] += 1
            continue

        totals["n_scored"]            += 1
        totals["sum_answer_at_k"]     += outcome["answer_at_k"]
        totals["sum_answer_top1"]     += outcome["answer_top1"]
        totals["sum_reciprocal_rank"] += outcome["reciprocal_rank"]
        totals["sum_precision_at_k"]  += outcome["precision_at_k"]
        totals["sum_recall_at_k"]     += outcome["recall_at_k"]
        totals["sum_latency_ms"]      += outcome["latency_ms"]

        if totals["n_scored"] % 50 == 0:
            totals["checkpoints"].append({
                "questions_done": resume_index + this_batch_n,
                "n_scored":        totals["n_scored"],
                **_compute_metrics(totals),
            })

    state["resume_index"] = resume_index + this_batch_n
    _save_checkpoint(path, state)

    return _summarize(state, retrieval_method, target=target_index,
                       this_batch_n=this_batch_n, top_k=top_k,
                       chunk_size=chunk_size, chunk_overlap=chunk_overlap)


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------

def _compute_metrics(totals: Dict[str, Any]) -> Dict[str, float]:
    n = totals["n_scored"]
    if n == 0:
        return {
            "answer_in_context_at_k": 0.0,
            "answer_in_top1":         0.0,
            "mrr":                    0.0,
            "precision_at_k":         0.0,
            "recall_at_k":            0.0,
            "avg_latency_ms":         0.0,
        }
    return {
        "answer_in_context_at_k": round(totals["sum_answer_at_k"]     / n, 4),
        "answer_in_top1":         round(totals["sum_answer_top1"]     / n, 4),
        "mrr":                    round(totals["sum_reciprocal_rank"] / n, 4),
        "precision_at_k":         round(totals["sum_precision_at_k"]  / n, 4),
        "recall_at_k":            round(totals["sum_recall_at_k"]     / n, 4),
        "avg_latency_ms":         round(totals["sum_latency_ms"]      / n, 2),
    }


def _summarize(state: Dict[str, Any], retrieval_method: str,
               target: Optional[int] = None,
               this_batch_n: int = 0,
               top_k: Optional[int] = None,
               chunk_size: Optional[int] = None,
               chunk_overlap: Optional[int] = None) -> Dict[str, Any]:
    totals = state["totals"]
    metrics = _compute_metrics(totals)

    return {
        "method":               retrieval_method,
        "resume_index":         state["resume_index"],
        "this_batch_questions": this_batch_n,
        "n_scored":             totals["n_scored"],
        "n_skipped_no_evidence": totals["n_skipped_no_evidence"],
        "top_k":                top_k,
        "chunk_size":           chunk_size,
        "chunk_overlap":        chunk_overlap,
        **metrics,
        "checkpoints":          totals["checkpoints"],
    }
