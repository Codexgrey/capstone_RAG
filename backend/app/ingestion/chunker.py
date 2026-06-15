"""
ingestion/chunker.py — Fixed-Size Word-Based Text Chunking

Splits extracted document text into overlapping chunks using a fixed
word-count window, aligning with the convention used by the vector
retrieval module (vector_retrieval/src/retrieval/vector_adapter.py).

Chunking strategy: fixed-size word windows with sentence-boundary snapping.
Page boundaries from [PAGE N] markers (inserted by parser.py) are respected —
chunks never span across page breaks.

Default chunk size: 200 words, overlap: 40 words.

NOTE: Adaptive/corpus-size-dependent chunking has been removed. All
documents are chunked with the same fixed parameters, ensuring Vector,
Keyword, and Hybrid retrieval operate over consistently-sized chunks
(modulo Vector's independent re-chunking of the same source text via its
own module chunker, which uses the same 200/40 convention — see
backend/app/retrieval/vector_adapter.py and
vector_retrieval/src/retrieval/vector_adapter.py).
"""

import re
from typing import List, Dict, Any, Tuple

# ── Fixed chunking parameters (words) ───────────────────────────────────────
DEFAULT_CHUNK_SIZE    = 200   # words per chunk
DEFAULT_CHUNK_OVERLAP = 40    # words shared between adjacent chunks


def chunk_text(
    text:           str,
    document_id:    str,
    source_name:    str,
    chunk_size:     int = DEFAULT_CHUNK_SIZE,
    chunk_overlap:  int = DEFAULT_CHUNK_OVERLAP,
) -> List[Dict[str, Any]]:
    """
    Chunk extracted document text into overlapping word-based segments.

    Returns list of chunk dicts compatible with the shared chunk schema:
        chunk_id, document_id, source_name, text, page,
        start_char, end_char, start_word_index, end_word_index, word_count
    """
    if not text or not text.strip():
        raise ValueError("Cannot chunk empty text")

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0.")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap cannot be negative.")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size.")

    pages   = _split_by_pages(text)
    chunks  = []
    idx     = 0

    for page_num, page_text in pages:
        for (chunk_content, start_char, end_char,
             start_word_idx, end_word_idx) in _split_into_chunks(
            page_text, chunk_size, chunk_overlap
        ):
            if not chunk_content.strip():
                continue
            chunks.append({
                "chunk_id":          f"{document_id}-chunk-{idx + 1}",
                "document_id":       document_id,
                "source_name":       source_name,
                "text":              chunk_content.strip(),
                "page":              page_num,
                "start_char":        start_char,
                "end_char":          end_char,
                "start_word_index":  start_word_idx,
                "end_word_index":    end_word_idx,
                "word_count":        end_word_idx - start_word_idx,
            })
            idx += 1

    if not chunks:
        raise ValueError(f"No chunks produced from document: {source_name}")

    print(f"  Chunked '{source_name}' → {len(chunks)} chunks "
          f"(size={chunk_size} words, overlap={chunk_overlap} words)")
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
) -> List[Tuple[str, int, int, int, int]]:
    """
    Slide a fixed-size word window over text, snapping the end of each
    chunk to the nearest preceding sentence boundary where possible.

    Returns a list of tuples:
        (chunk_text, start_char, end_char, start_word_index, end_word_index)
    """
    # Tokenize into words while tracking each word's char span in `text`.
    word_spans = [(m.start(), m.end()) for m in re.finditer(r'\S+', text)]
    n_words = len(word_spans)

    if n_words == 0:
        return []

    if n_words <= chunk_size:
        start_char = word_spans[0][0]
        end_char   = word_spans[-1][1]
        return [(text[start_char:end_char], start_char, end_char, 0, n_words)]

    chunks = []
    start_word = 0
    step = chunk_size - chunk_overlap

    while start_word < n_words:
        end_word = min(start_word + chunk_size, n_words)

        if end_word < n_words:
            snapped = _find_sentence_boundary_word(
                text, word_spans, start_word, end_word, lookback=20
            )
            if snapped is not None:
                end_word = snapped

        start_char = word_spans[start_word][0]
        end_char   = word_spans[end_word - 1][1]
        chunks.append((text[start_char:end_char], start_char, end_char,
                       start_word, end_word))

        if end_word >= n_words:
            break

        next_start = end_word - chunk_overlap
        start_word = next_start if next_start > start_word else start_word + step

    return chunks


def _find_sentence_boundary_word(
    text: str,
    word_spans: List[Tuple[int, int]],
    start_word: int,
    end_word: int,
    lookback: int = 20,
) -> int:
    """
    Look backwards from `end_word` (exclusive) for the last word whose text
    ends in sentence-ending punctuation (. ! ?), within `lookback` words.

    Returns the word index to use as the new exclusive end boundary
    (i.e. the chunk includes words up to and including the sentence-ending
    word), or None if no such word is found.
    """
    search_start = max(start_word, end_word - lookback)
    for i in range(end_word - 1, search_start - 1, -1):
        word_start, word_end = word_spans[i]
        word_text = text[word_start:word_end]
        if word_text and word_text[-1] in ".!?":
            return i + 1  # exclusive end boundary
    return None
