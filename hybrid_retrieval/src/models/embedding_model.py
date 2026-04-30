from sentence_transformers import SentenceTransformer

def load_embedding_model(model_name='all-MiniLM-L6-v2'):
    """
    Loads the sentence-transformer embedding model for sementic search.
    Returns the model ready to encode text.
    """
    try:
        model = SentenceTransformer(model_name)
        print(f'Embedding model loaded: {model_name}')
        return model
    except Exception as e:
        raise RuntimeError(f'Failed to load embedding model {model_name}: {e}')
