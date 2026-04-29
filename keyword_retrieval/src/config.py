"""
config
==========
Central configuration for the Keyword Retrieval RAG pipeline.
This file contains all the important settings and parameters for the system,
such as API keys, model names, storage paths, and default values for chunking and retrieval.
"""

# =============================================================================
# GROQ API — used for query normalisation (Step 7) and answer generation (Step 10)
# =============================================================================

import os
from dotenv import load_dotenv

load_dotenv()  
GROQ_API_KEY      = os.environ.get("GROQ_API_KEY", "your_groq_api_key_here")

# Model used to extract keywords from the user query (Step 7)
QUERY_MODEL_NAME  = "llama-3.1-8b-instant"

# Model used to generate the final answer (Step 10)
GENERATOR_MODEL   = "llama-3.1-8b-instant"

# Maximum tokens the generator can produce in one response
MAX_NEW_TOKENS    = 500

# Generation temperature — 0.0 = deterministic, 1.0 = creative
# Keep low (0.1) for factual RAG answers
TEMPERATURE       = 0.1


# =============================================================================
# STORAGE
# =============================================================================

# Folder where loaded documents are stored 
STORAGE_DIR       = r"C:\Users\DC\Desktop\keyword_RAG_01\tests"


# =============================================================================
# CHUNKING DEFAULTS
# =============================================================================

DEFAULT_CHUNK_SIZE    = 400   # words per chunk
DEFAULT_CHUNK_OVERLAP = 50    # words shared between adjacent chunks


# =============================================================================
# RETRIEVAL DEFAULTS
# =============================================================================

DEFAULT_TOP_K     = 5         # number of chunks to retrieve
