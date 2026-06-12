"""
indexing/bm25_store.py
=======================
Build, save, and load the BM25 scoring model.
Also exposes load_bm25() for the adapter to load all three
persisted objects (BM25 model, inverted index, chunk records) at once.
"""

import pickle
from pathlib import Path

from rank_bm25 import BM25Okapi


# =============================================================================
# BUILD BM25 MODEL
# =============================================================================

def build_bm25(tokenized_chunks: list[list[str]]) -> BM25Okapi:
    """
    Fit a BM25Okapi model on the tokenised chunks.

    Raises
    ------
    ValueError  if tokenized_chunks is empty.
    """
    if not tokenized_chunks:
        raise ValueError(
            "Cannot build BM25 model from an empty list. "
            "Make sure chunking and tokenising have been completed first."
        )

    bm25 = BM25Okapi(tokenized_chunks)
    print(f"  [BM25] Model built over {len(tokenized_chunks)} chunks.")
    return bm25


# =============================================================================
# SAVE AND LOAD — single-object helpers (used internally)
# =============================================================================

def save_bm25(bm25: BM25Okapi, path: str | Path) -> None:
    """Save the BM25 model to disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(bm25, f)
    print(f"  [BM25] Model saved to: {path}")


# =============================================================================
# LOAD ALL — called by keyword_adapter._load_state()
# =============================================================================

def load_bm25(
    bm25_path:   str | Path,
    index_path:  str | Path,
    chunks_path: str | Path,
) -> tuple:
    """
    Load the BM25 model, inverted index, and chunk records from disk.

    Called by keyword_adapter._load_state() to restore all module state
    in one call. Mirrors the pattern in vector_adapter's load_index().

    Parameters
    ----------
    bm25_path   : path to the saved BM25 pickle
    index_path  : path to the saved inverted index pickle
    chunks_path : path to the saved chunk records pickle

    Returns
    -------
    (bm25, inverted_index, chunk_records)

    Raises
    ------
    FileNotFoundError  if any of the three files do not exist.
    """
    for path in (bm25_path, index_path, chunks_path):
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(
                f"Keyword index file not found: {p}\n"
                "Call ingest(file_paths) before retrieve()."
            )

    with open(bm25_path,   "rb") as f: bm25           = pickle.load(f)
    with open(index_path,  "rb") as f: inverted_index = pickle.load(f)
    with open(chunks_path, "rb") as f: chunk_records  = pickle.load(f)

    print(f"  [BM25] State loaded — {len(chunk_records)} chunks, "
          f"{len(inverted_index):,} index terms.")

    return bm25, inverted_index, chunk_records