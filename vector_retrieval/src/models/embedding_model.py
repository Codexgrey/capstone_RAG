"""
src/models/embedding_model.py
Interface to the sentence-transformers embedding library.
Loads the selected embedding model and encodes text chunks into vectors.
"""

import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List


def load_embedding_model(model_name: str = 'all-MiniLM-L6-v2') -> SentenceTransformer:
    """
    Load a pre-trained SentenceTransformer embedding model.

    Args:
        model_name: HuggingFace model identifier.

    Returns:
        Loaded SentenceTransformer model instance.

    Raises:
        RuntimeError: If the model fails to load.
    """
    try:
        model = SentenceTransformer(model_name)
        print(f'Embedding model loaded: {model_name}')
        return model
    except Exception as e:
        raise RuntimeError(f'Failed to load embedding model {model_name}: {e}')


def encode_chunks(model: SentenceTransformer, chunk_texts: List[str]) -> np.ndarray:
    """
    Encode a list of text chunks into float32 embedding vectors.

    Args:
        model:       Loaded SentenceTransformer model.
        chunk_texts: List of chunk text strings to encode.

    Returns:
        NumPy array of shape (num_chunks, embedding_dim), dtype float32.
    """
    embeddings = model.encode(chunk_texts, convert_to_numpy=True)
    return np.array(embeddings, dtype='float32')
