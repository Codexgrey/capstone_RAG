"""
keyword_model
=======================
BM25-based keyword extraction and retrieval model.
"""

from rank_bm25 import BM25Okapi


class KeywordModel:
    """Wraps BM25Okapi and exposes a consistent API."""

    def __init__(self):
        self._bm25: BM25Okapi | None = None
        self._built = False

    def build(self, tokenized_chunks: list[list[str]]) -> None:
        """
        Fit the BM25 model on a list of pre-tokenised chunk token lists.

        """
        if not tokenized_chunks:
            raise ValueError("tokenized_chunks must not be empty.")
        self._bm25  = BM25Okapi(tokenized_chunks)
        self._built = True
        print(f"BM25 model built over {len(tokenized_chunks)} chunks.")

    def get_scores(self, query_tokens: list[str]) -> list[float]:
        """Return a BM25 score for every indexed chunk."""
        self._check_built()
        return self._bm25.get_scores(query_tokens)

    def _check_built(self):
        if not self._built:
            raise RuntimeError("Model not built yet. Call build() first.")
