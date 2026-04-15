import re
from typing import List, Dict, Any

def chunk_text(
    text: str,
    document_id: str,
    source_name: str,
    chunk_size: int = 500, # chunk size is 500
    chunk_overlap: int = 50, # overlap beteen chunk is 50
) -> List[Dict[str, Any]]: # specifies the return type

    if not text or not text.strip():
        raise ValueError("Cannot chunk empty text")

    # parse the page markers [PAGE #]
    pages = _split_by_pages(text)

    chunks = []
    chunk_index = 0

    for page_num, page_text in pages:
        page_chunks = _split_into_chunks(page_text, chunk_size, chunk_overlap)

        for chunk_text_content, start_char, end_char in page_chunks:
            if not chunk_text_content.strip():
                continue

            chunk_id = f"{document_id}_chunk_{chunk_index:04d}"

            chunks.append({
                "chunk_id":    chunk_id,
                "document_id": document_id,
                "source_name": source_name,
                "text": chunk_text_content.strip(),
                "page": page_num,
                "start_char": start_char,
                "end_char": end_char,
            })

            chunk_index += 1

    if not chunks:
        raise ValueError(f"No chunks produced from document: {source_name}")

    print(f"Chunked '{source_name}' -> {len(chunks)} chunks")
    return chunks

def _split_into_chunks(text: str, chunk_size: int, chunk_overlap: int):
    chunks = []
    start = 0
    text_length = len(text)

    # if text is smaller than chunk_size — return as single chunk
    if text_length <= chunk_size:
        return [(text, 0, text_length)]

    while start < text_length:
        end = min(start + chunk_size, text_length)

        if end < text_length:
            boundary = _find_sentence_boundary(text, end, lookback=100)
            if boundary:
                end = boundary

        chunk = text[start:end]
        chunks.append((chunk, start, end))

        next_start = end - chunk_overlap

        # safety guard — always move forward, never backwards
        if next_start <= start:
            next_start = start + chunk_size

        start = next_start

    return chunks
    # split a block of text into overlapping character-based chunks
    # tries to split at . ! ? or whitespace

    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = min(start + chunk_size, text_length)

        if end < text_length:
            # look for sentence - ending punctuation (? ! .)
            boundary = _find_sentence_boundary(text, end, lookback=100)
            if boundary:
                end = boundary

        chunk = text[start:end]
        chunks.append((chunk, start, end))

        # move the start forward
        start = end - chunk_overlap
        if start >= end:
            break

    return chunks

def _split_by_pages(text: str):
    # split text by page # markers inserted by parser
    page_pattern = re.compile(r'\[PAGE (\d+)\]', re.IGNORECASE)
    parts = page_pattern.split(text)

    if len(parts) == 1:
        return [(1, text)]
    
    pages = []

    i = 1
    while i < len(parts) - 1:
        try:
            page_num = int(parts[i])
            page_content = parts[i + 1]
            pages.append((page_num, page_content))
            i += 2
        except (ValueError, IndexError):
            i += 1
    
    return pages if pages else [(1, text)]

def _find_sentence_boundary(text: str, position: int, lookback: int = 100) -> int:
    # look backwards from position to find the last sentence end

    search_start = max(0, position - lookback)
    segment = text[search_start:position]

    # find the last occurence of sentence ending punction
    last_boundary = 0
    for i, char in enumerate(segment):
        if char in '.!?':
            last_boundary = search_start + i + 1

    return last_boundary if last_boundary > search_start else 0