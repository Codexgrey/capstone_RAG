"""
ingestion/chunker.py — Text Chunking

Splits extracted document text into overlapping character-based chunks.
Respects page boundaries from the [PAGE N] markers inserted by parser.py.

Used by upload.py after extract_text().
"""

import re
from typing import List, Dict, Any, Tuple


def chunk_text(
    text:           str,
    document_id:    str,
    source_name:    str,
    chunk_size:     int = 500,
    chunk_overlap:  int = 50,
) -> List[Dict[str, Any]]:
    """
    Chunk extracted document text into overlapping segments.

    Returns list of chunk dicts compatible with the shared chunk schema:
        chunk_id, document_id, source_name, text, page, start_char, end_char
    """
    if not text or not text.strip():
        raise ValueError("Cannot chunk empty text")

    pages   = _split_by_pages(text)
    chunks  = []
    idx     = 0

    for page_num, page_text in pages:
        for chunk_content, start_char, end_char in _split_into_chunks(page_text, chunk_size, chunk_overlap):
            if not chunk_content.strip():
                continue
            chunks.append({
                "chunk_id":    f"{document_id}-chunk-{idx + 1}",
                "document_id": document_id,
                "source_name": source_name,
                "text":        chunk_content.strip(),
                "page":        page_num,
                "start_char":  start_char,
                "end_char":    end_char,
            })
            idx += 1

    if not chunks:
        raise ValueError(f"No chunks produced from document: {source_name}")

    print(f"  Chunked '{source_name}' → {len(chunks)} chunks")
    return chunks


# ── Helpers ────────────────────────────────────────────────────────────────────

def _split_by_pages(text: str) -> List[Tuple[int, str]]:
    """Split text on [PAGE N] markers inserted by parser.py."""
    parts = re.compile(r'\[PAGE (\d+)\]', re.IGNORECASE).split(text)
    if len(parts) == 1:
        return [(1, text)]
    pages, i = [], 1
    while i < len(parts) - 1:
        try:
            pages.append((int(parts[i]), parts[i + 1]))
            i += 2
        except (ValueError, IndexError):
            i += 1
    return pages if pages else [(1, text)]


def _split_into_chunks(text: str, chunk_size: int, chunk_overlap: int) -> List[Tuple[str, int, int]]:
    """Slide a window over text, splitting at sentence boundaries when possible."""
    if len(text) <= chunk_size:
        return [(text, 0, len(text))]

    chunks, start = [], 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            boundary = _find_sentence_boundary(text, end, lookback=100)
            if boundary:
                end = boundary
        chunks.append((text[start:end], start, end))
        next_start = end - chunk_overlap
        start = next_start if next_start > start else start + chunk_size
    return chunks


def _find_sentence_boundary(text: str, position: int, lookback: int = 100) -> int:
    """Find the last sentence-ending punctuation before position."""
    search_start = max(0, position - lookback)
    last = 0
    for i, ch in enumerate(text[search_start:position]):
        if ch in ".!?":
            last = search_start + i + 1
    return last if last > search_start else 0
