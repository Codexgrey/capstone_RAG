"""
ingestion/chunker.py — Adaptive Text Chunking

Splits extracted document text into overlapping chunks.
Chunk size and overlap adapt based on the total token volume across all
currently ingested documents, so that retrieval precision scales with corpus
size rather than using a fixed size that works poorly at either extreme.

Chunking strategy: fixed-size character windows with sentence-boundary snapping.
Page boundaries from [PAGE N] markers (inserted by parser.py) are respected —
chunks never span across page breaks.

Adaptive sizing table (approximate word tokens, 1 token ≈ 4 chars):
  ≤ 10 k tokens  → chunk 300  / overlap 40   (small corpus, high precision)
  ≤ 50 k tokens  → chunk 400  / overlap 60   (medium corpus, balanced)
  ≤ 150 k tokens → chunk 500  / overlap 80   (large corpus, more context)
  >  150 k tokens → chunk 650  / overlap 100  (very large corpus, broad context)
"""

import re
from typing import List, Dict, Any, Tuple


# ── Adaptive sizing thresholds (total chars across all ingested docs) ──────────
# 1 avg token ≈ 4 chars; thresholds in character counts
_TIERS = [
    (40_000,   300,  40),   # ≤ ~10 k tokens
    (200_000,  400,  60),   # ≤ ~50 k tokens
    (600_000,  500,  80),   # ≤ ~150 k tokens
    (float("inf"), 650, 100),  # > ~150 k tokens
]


def get_adaptive_chunk_params(total_ingested_chars: int) -> Tuple[int, int]:
    """
    Return (chunk_size, chunk_overlap) tuned to the total character volume
    of all currently ingested documents.

    Smaller corpora get smaller, more precise chunks.
    Larger corpora get broader chunks to keep retrieval tractable.
    """
    for threshold, size, overlap in _TIERS:
        if total_ingested_chars <= threshold:
            return size, overlap
    return 650, 100  # fallback


def _get_total_ingested_chars() -> int:
    """Query PostgreSQL for the total character count across all completed chunks."""
    try:
        from app.config.database import SessionLocal
        from app.models.db_models import DocumentChunk
        db = SessionLocal()
        try:
            # Sum text lengths of all stored chunks
            chunks = db.query(DocumentChunk.text).all()
            return sum(len(c.text) for c in chunks)
        finally:
            db.close()
    except Exception:
        return 0  # fall back to smallest tier if DB not available


def chunk_text(
    text:           str,
    document_id:    str,
    source_name:    str,
    chunk_size:     int = None,   # None → auto-select based on corpus size
    chunk_overlap:  int = None,   # None → auto-select based on corpus size
) -> List[Dict[str, Any]]:
    """
    Chunk extracted document text into overlapping segments.

    When chunk_size / chunk_overlap are None (the default), sizes are chosen
    adaptively based on the total volume of already-ingested text so that
    chunk granularity scales with corpus size.

    Returns list of chunk dicts compatible with the shared chunk schema:
        chunk_id, document_id, source_name, text, page, start_char, end_char
    """
    if not text or not text.strip():
        raise ValueError("Cannot chunk empty text")

    # Determine chunk parameters
    if chunk_size is None or chunk_overlap is None:
        total_chars = _get_total_ingested_chars()
        auto_size, auto_overlap = get_adaptive_chunk_params(total_chars)
        chunk_size    = chunk_size    if chunk_size    is not None else auto_size
        chunk_overlap = chunk_overlap if chunk_overlap is not None else auto_overlap

    pages   = _split_by_pages(text)
    chunks  = []
    idx     = 0

    for page_num, page_text in pages:
        for chunk_content, start_char, end_char in _split_into_chunks(
            page_text, chunk_size, chunk_overlap
        ):
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

    print(f"  Chunked '{source_name}' → {len(chunks)} chunks "
          f"(size={chunk_size}, overlap={chunk_overlap})")
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


def _split_into_chunks(
    text: str, chunk_size: int, chunk_overlap: int
) -> List[Tuple[str, int, int]]:
    """Slide a window over text, snapping splits to sentence boundaries."""
    if len(text) <= chunk_size:
        return [(text, 0, len(text))]

    chunks, start = [], 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            boundary = _find_sentence_boundary(text, end, lookback=120)
            if boundary:
                end = boundary
        chunks.append((text[start:end], start, end))
        next_start = end - chunk_overlap
        start = next_start if next_start > start else start + chunk_size
    return chunks


def _find_sentence_boundary(text: str, position: int, lookback: int = 120) -> int:
    """Find the last sentence-ending punctuation before position."""
    search_start = max(0, position - lookback)
    last = 0
    for i, ch in enumerate(text[search_start:position]):
        if ch in ".!?":
            last = search_start + i + 1
    return last if last > search_start else 0
