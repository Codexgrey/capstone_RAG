"""
src/utils/chunker.py
Helper utility — splits a document into overlapping word-level chunks with metadata.
"""

from typing import List, Dict, Any


def chunk_text_with_metadata(
    text: str,
    chunk_size: int = 150,
    overlap: int = 30,
    document_title: str = 'Untitled',
    source: str = 'unknown',
    document_id: str = 'doc-000'
) -> List[Dict[str, Any]]:
    """
    Split text into fixed-size word chunks with overlap, attaching document
    metadata to each chunk record.

    Args:
        text:           Raw document text to split.
        chunk_size:     Number of words per chunk.
        overlap:        Number of words shared between adjacent chunks.
        document_title: Human-readable title of the source document.
        source:         File path or identifier of the source document.
        document_id:    Unique identifier for the source document.

    Returns:
        List of chunk dicts, each containing:
            document_id, document_title, source, chunk_id, chunk_index,
            start_word_index, end_word_index, word_count, overlap, text.

    Raises:
        ValueError: If chunk_size <= 0, overlap < 0, or overlap >= chunk_size.
    """
    if chunk_size <= 0:
        raise ValueError('chunk_size must be greater than 0.')
    if overlap < 0:
        raise ValueError('overlap cannot be negative.')
    if overlap >= chunk_size:
        raise ValueError('overlap must be smaller than chunk_size.')

    words = text.split()
    chunk_records = []
    step = chunk_size - overlap

    for start_idx in range(0, len(words), step):
        end_idx = start_idx + chunk_size
        chunk_words = words[start_idx:end_idx]

        if not chunk_words:
            continue

        chunk_text = ' '.join(chunk_words)
        chunk_index = len(chunk_records)

        chunk_records.append({
            'document_id': document_id,
            'document_title': document_title,
            'source': source,
            'chunk_id': f'{document_id}-chunk-{chunk_index + 1}',
            'chunk_index': chunk_index,
            'start_word_index': start_idx,
            'end_word_index': min(end_idx, len(words)),
            'word_count': len(chunk_words),
            'overlap': overlap,
            'text': chunk_text
        })

        if end_idx >= len(words):
            break

    return chunk_records
