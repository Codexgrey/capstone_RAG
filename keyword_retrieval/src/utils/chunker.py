
# =============================================================================
# STEP 4 — CHUNK TEXT WITH METADATA
# =============================================================================

def chunk_text_with_metadata(
    text:           str,
    chunk_size:     int = 400,
    overlap:        int = 50,
    document_title: str = "Untitled",
    source:         str = "unknown",
    document_id:    str = "doc-000",
    lang_code:      str = "en",
) -> list[dict]:
   
    # --- Validate inputs ---
    if chunk_size <= 0:
        raise ValueError(
            f"chunk_size must be greater than 0. Got: {chunk_size}"
        )
    if overlap < 0:
        raise ValueError(
            f"overlap cannot be negative. Got: {overlap}"
        )
    if overlap >= chunk_size:
        raise ValueError(
            f"overlap ({overlap}) must be smaller than chunk_size ({chunk_size})."
        )

    # --- Split text into words ---
    words = text.split()

    if not words:
        return []   # empty document — return empty list

    # --- Slide the window ---
    chunk_records = []
    step          = chunk_size - overlap   # how far to advance each time

    for start_idx in range(0, len(words), step):

        end_idx     = start_idx + chunk_size
        chunk_words = words[start_idx:end_idx]

        # Skip if somehow empty (edge case)
        if not chunk_words:
            continue

        chunk_index = len(chunk_records)   # 0-based index of this chunk

        # Build the chunk dict
        chunk = {
            "document_id":      document_id,
            "document_title":   document_title,
            "source":           source,
            "chunk_id":         f"{document_id}-chunk-{chunk_index + 1}",
            "chunk_index":      chunk_index,
            "start_word_index": start_idx,
            "end_word_index":   min(end_idx, len(words)),
            "word_count":       len(chunk_words),
            "overlap":          overlap,
            "lang_code":        lang_code,
            "text":             " ".join(chunk_words),
        }

        chunk_records.append(chunk)

        # Stop when we have reached the end of the document
        if end_idx >= len(words):
            break

    return chunk_records
