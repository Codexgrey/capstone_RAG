"""
generation/response_formatter.py
Formats the LLM answer into the team's shared answer_response contract.

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
    question: str = "",          # ← Collins contract requires this
) -> Dict[str, Any]:

    citations    = _build_citations(chunks)
    evidence     = _build_evidence(chunks)

    response = {
        "query":            question,          # ← contract field
        "answer":           answer,
        "evidence_used":    evidence,          # ← contract field
        "citations":        citations,
        "retrieval_method": retrieval_method,
        "latency_ms":       round(latency_ms, 2),
    }

    if session_id:
        response["session_id"] = session_id

    return response


def _build_citations(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Build citations matching Collins's answer_response contract:
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

        # Generate document_title from filename (Collins contract)
        doc_title = _clean_source_name(source_name)\
            .replace(".pdf",  "")\
            .replace(".txt",  "")\
            .replace(".docx", "")\
            .replace(".md",   "")\
            .replace("_",     " ")\
            .title()

        citations.append({
            # Collins contract fields
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
    Build evidence_used array per Collins's contract.
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