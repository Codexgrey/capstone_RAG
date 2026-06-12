"""
generation/response_formatter.py
Formats the LLM answer into the team's shared answer_response schema.

evaluation_metrics includes TriviaQA EM/F1 (scored on the fly via the
5,000-question bank) whenever the user's question matches a bank entry,
plus the existing retrieval quality metrics.
"""

from typing import List, Dict, Any, Optional
import re


def format_response(
    answer: str,
    chunks: List[Dict[str, Any]],
    retrieval_method: str = "none",
    latency_ms: float = 0.0,
    session_id: str = None,
    question: str = "",
) -> Dict[str, Any]:

    citations    = _build_citations(chunks)
    evidence     = _build_evidence(chunks)
    eval_metrics = _compute_evaluation_metrics(chunks, question, latency_ms, answer)

    response = {
        "query":              question,
        "answer":             answer,
        "evidence_used":      evidence,
        "citations":          citations,
        "retrieval_method":   retrieval_method,
        "latency_ms":         round(latency_ms, 2),
        "evaluation_metrics": eval_metrics,
    }

    if session_id:
        response["session_id"] = session_id

    return response


def _compute_evaluation_metrics(
    chunks: List[Dict[str, Any]],
    question: str = "",
    latency_ms: float = 0.0,
    answer: str = "",
) -> Dict[str, Any]:
    """
    Compute retrieval evaluation metrics from returned chunks,
    plus TriviaQA EM/F1 if the question is in the 5,000-question bank.
    """
    # ── Retrieval quality metrics ─────────────────────────────────────────
    if not chunks:
        base = {
            "top_score":        0.0,
            "avg_score":        0.0,
            "source_coverage":  0,
            "chunks_retrieved": 0,
            "precision_at_k":   0.0,
            "mrr":              0.0,
            "source_diversity": 0.0,
        }
    else:
        k = len(chunks)
        scores = []
        for c in chunks:
            raw = c.get("score", 0.0)
            try:
                s = float(raw)
            except (TypeError, ValueError):
                s = 0.0
            scores.append(max(0.0, min(s, 1.0)) if s <= 1.0 else min(s / (s + 10.0), 1.0))

        top_score = round(scores[0], 4) if scores else 0.0
        avg_score = round(sum(scores) / k, 4) if scores else 0.0

        unique_sources = {
            c.get("source_name") or c.get("source") or c.get("document_id", "unknown")
            for c in chunks
        }
        source_coverage = len(unique_sources)

        RELEVANCE_THRESHOLD = 0.40
        relevant_ranks  = [i + 1 for i, s in enumerate(scores) if s >= RELEVANCE_THRESHOLD]
        precision_at_k  = round(len(relevant_ranks) / k, 4) if k else 0.0
        mrr             = round(1.0 / relevant_ranks[0], 4) if relevant_ranks else 0.0
        source_diversity = round(source_coverage / k, 4) if k else 0.0

        base = {
            "top_score":        top_score,
            "avg_score":        avg_score,
            "source_coverage":  source_coverage,
            "chunks_retrieved": k,
            "precision_at_k":   precision_at_k,
            "mrr":              mrr,
            "source_diversity": source_diversity,
        }

    # ── TriviaQA EM/F1 (on-the-fly when question is in bank) ─────────────
    try:
        from app.evaluation.triviaqa_evaluator import score_if_in_bank
        triviaqa = score_if_in_bank(question, answer)
        if triviaqa:
            base.update(triviaqa)
    except ImportError:
        pass

    return base


def _build_citations(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    citations = []
    seen = set()

    for chunk in chunks:
        metadata    = chunk.get("metadata", {})
        source_name = _clean_source_name(chunk.get("source_name", "Unknown"))
        chunk_id    = chunk.get("chunk_id", "")
        page        = metadata.get("page", None)
        section     = metadata.get("section", None)

        key = f"{source_name}_{chunk_id}"
        if key in seen:
            continue
        seen.add(key)

        doc_title = (
            _clean_source_name(source_name)
            .replace(".pdf",  "")
            .replace(".txt",  "")
            .replace(".docx", "")
            .replace(".md",   "")
            .replace("_",     " ")
            .title()
        )

        citations.append({
            "chunk_id":       chunk_id,
            "document_title": doc_title,
            "source":         source_name,
            "file_type":      metadata.get("file_type", ""),
            "source_name":    source_name,
            "page":           page,
            "section":        section,
        })

    return citations


def _build_evidence(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        {
            "chunk_id":     chunk.get("chunk_id", ""),
            "contribution": (
                chunk.get("text", "")[:150] + "..."
                if len(chunk.get("text", "")) > 150
                else chunk.get("text", "")
            ),
        }
        for chunk in chunks
    ]


def format_error_response(error_message: str, retrieval_method: str = "none") -> Dict[str, Any]:
    return {
        "query":            "",
        "answer":           f"An error occurred: {error_message}",
        "evidence_used":    [],
        "citations":        [],
        "retrieval_method": retrieval_method,
        "latency_ms":       0.0,
    }


def _clean_source_name(name: str) -> str:
    name = name.replace("_ocr.txt", ".pdf").replace("_ocr", "")
    name = re.sub(r"^[0-9a-f]{8}_", "", name)
    return name
