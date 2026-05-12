"""
generation/response_formatter.py
Formats the LLM answer into the team's shared answer_response schema.

Matches answer_response.schema.json exactly:
{
    "query":            "What is RAG?",
    "answer":           "RAG stands for...",
    "evidence_used":    [...],
    "citations":        [...],
    "retrieval_method": "vector",
    "latency_ms":       320.5,
    "session_id":       "uuid..."
}
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
    eval_metrics = _compute_evaluation_metrics(chunks, question, latency_ms)

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
) -> Dict[str, Any]:
    """
    Compute retrieval evaluation metrics from the returned chunks.

    Metrics:
      - top_score        : similarity/BM25 score of rank-1 chunk  (float, 0–1 normalised)
      - avg_score        : mean score across all returned chunks
      - source_coverage  : number of unique documents represented
      - chunks_retrieved : total chunks returned (top-k)
      - precision_at_k   : fraction of chunks with score >= 0.40 (relevance proxy)
      - mrr              : reciprocal rank of the first "relevant" chunk (score >= 0.40)
      - source_diversity : source_coverage / chunks_retrieved  (0–1)

    All values are rounded to 4 decimal places where applicable.
    """
    if not chunks:
        return {
            "top_score":        0.0,
            "avg_score":        0.0,
            "source_coverage":  0,
            "chunks_retrieved": 0,
            "precision_at_k":   0.0,
            "mrr":              0.0,
            "source_diversity": 0.0,
        }

    k = len(chunks)
    scores = []
    for c in chunks:
        raw = c.get("score", 0.0)
        # BM25 scores can be large; normalise to 0-1 via sigmoid-like cap
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

    # Relevance proxy: score >= 0.40 counts as "relevant"
    RELEVANCE_THRESHOLD = 0.40
    relevant_ranks = [
        i + 1 for i, s in enumerate(scores) if s >= RELEVANCE_THRESHOLD
    ]
    precision_at_k = round(len(relevant_ranks) / k, 4) if k else 0.0
    mrr = round(1.0 / relevant_ranks[0], 4) if relevant_ranks else 0.0
    source_diversity = round(source_coverage / k, 4) if k else 0.0

    return {
        "top_score":        top_score,
        "avg_score":        avg_score,
        "source_coverage":  source_coverage,
        "chunks_retrieved": k,
        "precision_at_k":   precision_at_k,
        "mrr":              mrr,
        "source_diversity": source_diversity,
    }


def _build_citations(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Build citations matching project's answer_response schema:
    { chunk_id, document_title, source, file_type }

    Also keeps source_name and page so SourcesPanel.tsx works without changes.
    """
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

        # Generate document_title from filename (shared contract)
        doc_title = _clean_source_name(source_name)\
            .replace(".pdf",  "")\
            .replace(".txt",  "")\
            .replace(".docx", "")\
            .replace(".md",   "")\
            .replace("_",     " ")\
            .title()

        citations.append({
            # Shared contract fields
            "chunk_id":       chunk_id,
            "document_title": doc_title,
            "source":         source_name,
            "file_type":      metadata.get("file_type", ""),
            # kept for SourcesPanel.tsx
            "source_name":    source_name,
            "page":           page,
            "section":        section,
        })

    return citations


def _build_evidence(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Build evidence_used array per project shared contract.
    Shows which chunks contributed to the answer and a preview.
    """
    return [
        {
            "chunk_id":     chunk.get("chunk_id", ""),
            "contribution": chunk.get("text", "")[:150] + "..."
                            if len(chunk.get("text", "")) > 150
                            else chunk.get("text", ""),
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
    """Remove OCR suffix and UUID prefix from filenames."""
    name = name.replace("_ocr.txt", ".pdf").replace("_ocr", "")
    name = re.sub(r'^[0-9a-f]{8}_', '', name)
    return name