
import pickle
from pathlib import Path

from rank_bm25 import BM25Okapi


# =============================================================================
# STEP 6 — BUILD BM25 MODEL
# =============================================================================

def build_bm25(tokenized_chunks: list[list[str]]) -> BM25Okapi:
    
    if not tokenized_chunks:
        raise ValueError(
            "Cannot build BM25 model from an empty list. "
            "Make sure Step 4 (chunking) and Step 5 (tokenising) "
            "have been completed first."
        )

    bm25 = BM25Okapi(tokenized_chunks)
    print(f"  [BM25] Model built over {len(tokenized_chunks)} chunks.")
    return bm25


# =============================================================================
# SAVE AND LOAD — persist the model to avoid rebuilding on every run
# =============================================================================

def save_bm25(bm25: BM25Okapi, path: str | Path) -> None:
    
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "wb") as f:
        pickle.dump(bm25, f)

    print(f"  [BM25] Model saved to: {path}")


def load_bm25(path: str | Path) -> BM25Okapi:
   
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"BM25 model file not found: {path}\n"
            "Run Step 6 to build and save the model first."
        )

    with open(path, "rb") as f:
        bm25 = pickle.load(f)

    print(f"  [BM25] Model loaded from: {path}")
    return bm25