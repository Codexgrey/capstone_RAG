"""
indexing/indexer.py
====================
Builds the positional inverted index and exposes the full ingestion
pipeline used by the adapter.
"""

import collections
import datetime
import pickle
from pathlib import Path


# =============================================================================
# BUILD INVERTED INDEX
# =============================================================================

def build_inverted_index(
    chunk_records:    list[dict],
    tokenized_chunks: list[list[str]],
) -> dict:
    """
    Build a positional inverted index from tokenised chunks.

    Parameters
    ----------
    chunk_records    : list[dict]   chunk metadata dicts from chunker.py
    tokenized_chunks : list[list[str]]   token lists from preprocess.py

    Returns
    -------
    dict  —  inverted index  { term: { doc_freq, postings: [...] } }

    Raises
    ------
    ValueError  if chunk_records and tokenized_chunks have different lengths.
    """
    if len(chunk_records) != len(tokenized_chunks):
        raise ValueError(
            f"chunk_records ({len(chunk_records)}) and "
            f"tokenized_chunks ({len(tokenized_chunks)}) "
            f"must have the same length."
        )

    index = {}

    for chunk_idx, (chunk, tokens) in enumerate(zip(chunk_records, tokenized_chunks)):
        chunk_id       = chunk["chunk_id"]
        term_positions = collections.defaultdict(list)

        for position, token in enumerate(tokens):
            term_positions[token].append(position)

        for term, positions in term_positions.items():
            if term not in index:
                index[term] = {"doc_freq": 0, "postings": []}

            index[term]["doc_freq"] += 1
            index[term]["postings"].append({
                "chunk_id":  chunk_id,
                "chunk_idx": chunk_idx,
                "tf":        len(positions),
                "positions": positions,
            })

    return index


# =============================================================================
# BUILD PIPELINE — called by keyword_adapter.ingest()
# =============================================================================

def build_pipeline(
    file_paths:    list[str],
    chunk_size:    int = 300,
    chunk_overlap: int = 50,
    index_path:    str = "keyword_index.pkl",
    bm25_path:     str = "keyword_bm25.pkl",
    chunks_path:   str = "keyword_chunks.pkl",
) -> tuple[list[dict], dict]:
    """
    Full ingestion pipeline: load → clean → detect language →
    chunk → tokenise → build inverted index + BM25 → persist to disk.

    Called by keyword_adapter.ingest(). All heavy work lives here;
    the adapter stays thin.

    Parameters
    ----------
    file_paths    : list of absolute/relative file paths to ingest
    chunk_size    : words per chunk (default: 300)
    chunk_overlap : overlapping words between adjacent chunks (default: 50)
    index_path    : where to save the inverted index pickle
    bm25_path     : where to save the BM25 model pickle
    chunks_path   : where to save the chunk records pickle

    Returns
    -------
    (chunk_records, inverted_index)
        chunk_records  : list of chunk dicts (with metadata attached)
        inverted_index : the built inverted index dict

    Raises
    ------
    ValueError  if no files could be successfully processed.
    """
    from utils.loader      import _FORMAT_LOADERS
    from utils.chunker     import chunk_text_with_metadata
    from indexing.bm25_store import build_bm25, save_bm25
    from preprocessing.preprocess import (
        clean_text, detect_language, tokenize_chunk
    )

    all_chunk_records    = []
    all_tokenized_chunks = []
    uploaded_at          = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    docs_ingested        = 0

    for file_path in file_paths:
        path = Path(file_path)
        ext  = path.suffix.lower()

        if ext not in _FORMAT_LOADERS:
            continue

        try:
            text = _FORMAT_LOADERS[ext](path)
        except Exception:
            continue

        if not text or not text.strip():
            continue

        # Clean → detect language → chunk → tokenise
        cleaned              = clean_text(text)
        lang_code, nltk_lang = detect_language(cleaned)
        doc_num              = docs_ingested + 1
        doc_id               = f"doc-{doc_num:03d}"
        doc_title            = path.stem.replace("_", " ").replace("-", " ").title()

        chunks = chunk_text_with_metadata(
            cleaned,
            chunk_size    = chunk_size,
            overlap       = chunk_overlap,
            document_title= doc_title,
            source        = path.name,
            document_id   = doc_id,
            lang_code     = lang_code,
        )

        # Attach metadata required by the contract
        for chunk in chunks:
            chunk["metadata"] = {
                "file_name":    path.name,
                "file_type":    ext.lstrip("."),
                "file_size_kb": round(path.stat().st_size / 1024, 2),
                "uploaded_at":  uploaded_at,
            }

        tokenized = [tokenize_chunk(c["text"], nltk_lang) for c in chunks]

        all_chunk_records.extend(chunks)
        all_tokenized_chunks.extend(tokenized)
        docs_ingested += 1

    if docs_ingested == 0:
        raise ValueError("No supported files could be processed.")

    # Build index and BM25
    inverted_index = build_inverted_index(all_chunk_records, all_tokenized_chunks)
    bm25           = build_bm25(all_tokenized_chunks)

    # Persist all three to disk
    _save_pickle(inverted_index, index_path)
    save_bm25(bm25, bm25_path)
    _save_pickle(all_chunk_records, chunks_path)

    return all_chunk_records, inverted_index


# =============================================================================
# UTILITY HELPERS
# =============================================================================

def _save_pickle(obj, path: str | Path) -> None:
    """Save any object to disk as a pickle file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def inspect_term(term: str, index: dict, stemmer=None) -> None:
    """Pretty-print an index entry for a single term (for debugging)."""
    key   = stemmer.stem(term.lower()) if stemmer else term.lower()
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