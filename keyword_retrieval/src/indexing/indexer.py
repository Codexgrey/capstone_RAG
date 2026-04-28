
import collections


# =============================================================================
# STEP 6 — BUILD INVERTED INDEX
# =============================================================================

def build_inverted_index(
    chunk_records:    list[dict],
    tokenized_chunks: list[list[str]],
) -> dict:
    """
    Build a positional inverted index from tokenised chunks.

    How it works:
        For every chunk, iterate through its tokens.
        Record the position of each token inside that chunk.
        Store the result under the token's entry in the index.

        After processing all chunks, every term in the index
        has a list of postings — one per chunk that contains it.

    Parameters
    ----------
    chunk_records    : list[dict]
        The chunk metadata dicts produced by chunker.
        Used to retrieve chunk_id for each posting.

    tokenized_chunks : list[list[str]]
        The token lists produced by preprocess.py (Step 5).
        One list of tokens per chunk, same order as chunk_records.


    Raises
    ------
    ValueError
        If chunk_records and tokenized_chunks have different lengths.

    Example
    -------
        >>> chunks  = [{"chunk_id": "doc-001-chunk-1", ...}]
        >>> tokens  = [["cat", "chase", "mouse"]]
        >>> index   = build_inverted_index(chunks, tokens)
        >>> index["cat"]["doc_freq"]
        1
        >>> index["cat"]["postings"][0]["chunk_id"]
        'doc-001-chunk-1'
    """
    # --- Validate inputs ---
    if len(chunk_records) != len(tokenized_chunks):
        raise ValueError(
            f"chunk_records ({len(chunk_records)}) and "
            f"tokenized_chunks ({len(tokenized_chunks)}) "
            f"must have the same length."
        )

    index = {}   # this is the inverted index we are building

    # --- Process each chunk ---
    for chunk_idx, (chunk, tokens) in enumerate(zip(chunk_records, tokenized_chunks)):

        chunk_id = chunk["chunk_id"]

        # Collect the positions of every token inside this chunk.
        # e.g. tokens = ["cat", "chase", "cat"]
        #      term_positions = {"cat": [0, 2], "chase": [1]}
        term_positions = collections.defaultdict(list)
        for position, token in enumerate(tokens):
            term_positions[token].append(position)

        # Add each term's posting to the index
        for term, positions in term_positions.items():

            # Create a new entry if this term has not been seen before
            if term not in index:
                index[term] = {
                    "doc_freq": 0,       # number of chunks containing this term
                    "postings": [],      # one posting per chunk
                }

            # Increment document frequency
            index[term]["doc_freq"] += 1

            # Add a posting for this chunk
            index[term]["postings"].append({
                "chunk_id":  chunk_id,
                "chunk_idx": chunk_idx,
                "tf":        len(positions),   # term frequency in this chunk
                "positions": positions,        # token positions (for phrase search)
            })

    return index


# =============================================================================
# UTILITY — inspect a single term in the index (useful for debugging)
# =============================================================================

def inspect_term(term: str, index: dict, stemmer=None) -> None:
   
    # Stem the term if a stemmer is provided (index stores stemmed tokens)
    key = stemmer.stem(term.lower()) if stemmer else term.lower()
    entry = index.get(key)

    if not entry:
        print(f"  Term '{term}' (key: '{key}') — not found in index.")
        return

    print(f"  Term     : '{term}'  (key → '{key}')")
    print(f"  doc_freq : {entry['doc_freq']}  "
          f"(appears in {entry['doc_freq']} chunk(s))")

    for posting in entry["postings"]:
        pos_preview = posting["positions"][:6]
        ellipsis    = "..." if len(posting["positions"]) > 6 else ""
        print(f"    chunk_id={posting['chunk_id']}  "
              f"tf={posting['tf']}  "
              f"positions={pos_preview}{ellipsis}")