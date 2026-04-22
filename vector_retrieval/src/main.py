"""
src/main.py
Local entry point for the Vector Retrieval module.

Run from the vector_retrieval/ root:
    python -m src.main

Pipeline (mirrors notebook Sections 2 - 14, then Evaluation):
    Document ingestion loop   (Sections 2, 3, 5)
    Embed all chunks          (Section 7)
    Build + save FAISS index  (Section 8)
    Retriever                 (Section 9)    <- timed for latency
    Prompt builder            (Section 12)
    Generator (Groq)          (Section 13)
    Formatted report          (Section 14)
    Evaluation report         (Evaluation)

To ingest documents:
    - Place files in the tests/ folder, OR
    - Edit DOCUMENT_FOLDER below to point at any folder of supported files.

Supported file types: .txt, .pdf, .docx, .md
"""

import os
from src.utils.loader import get_files_from_folder
from src.indexing.indexer import build_pipeline, INDEX_SAVE_PATH, CHUNKS_SAVE_PATH
from src.indexing.vector_store import load_index
from src.retrieval.retriever import retrieve
from src.utils.prompts import build_prompt
from src.utils.response_printer import print_report
from src.evaluation.evaluate import Timer, run_evaluation
from groq import Groq
from dotenv import load_dotenv


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------

DOCUMENT_FOLDER      = 'tests'          # folder scanned for documents
CHUNK_SIZE           = 300
CHUNK_OVERLAP        = 50
TOP_K                = 3

REBUILD_INDEX        = True             # set False to reload a saved index instead

load_dotenv()
GROQ_API_KEY         = os.environ.get('GROQ_API_KEY', 'YOUR_GROQ_API_KEY_HERE')
GENERATOR_MODEL_NAME = 'llama-3.1-8b-instant'

QUERY = """
    What kind of neural network was used before transformers? 
    Summarise how The dimensionality of these embeddings affects retrieval precision, 
    while the chunk size impacts semantic coherence.
"""


# ---------------------------------------------------------------------------
# generator
# ---------------------------------------------------------------------------

def generate(prompt: str, client: Groq, max_new_tokens: int = 500) -> str:
    """Send a prompt to the Groq LLM and return the generated text."""
    response = client.chat.completions.create(
        model=GENERATOR_MODEL_NAME,
        messages=[{'role': 'user', 'content': prompt}],
        max_tokens=max_new_tokens,
        temperature=0.1,
    )
    return response.choices[0].message.content


# ---------------------------------------------------------------------------
# main pipeline
# ---------------------------------------------------------------------------

def run(query: str) -> None:
    """
    Execute the full vector retrieval RAG pipeline for a given query.

    Args:
        query: Natural language question to answer from the ingested documents.
    """
    # -----------------------------------------------------------------------
    # Sections 2, 3, 5 — scan folder, ingest all documents, build index
    # -----------------------------------------------------------------------
    document_paths = get_files_from_folder(DOCUMENT_FOLDER)

    if not document_paths:
        print(f'No supported documents found in "{DOCUMENT_FOLDER}". '
              f'Add .txt, .pdf, .docx, or .md files and retry.')
        return

    if REBUILD_INDEX:
        index, all_chunk_records, embedding_model = build_pipeline(
            document_paths=document_paths,
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            index_path=INDEX_SAVE_PATH,
            chunks_path=CHUNKS_SAVE_PATH,
        )
    else:
        # reload a previously saved index — skips re-embedding
        from src.models.embedding_model import load_embedding_model
        embedding_model = load_embedding_model()
        index, all_chunk_records = load_index(INDEX_SAVE_PATH, CHUNKS_SAVE_PATH)

    # -----------------------------------------------------------------------
    # section 9 — retriever (timed for latency)
    # -----------------------------------------------------------------------
    with Timer() as retrieval_timer:
        retrieved_results = retrieve(
            query, embedding_model, index, all_chunk_records, top_k=TOP_K
        )

    # -----------------------------------------------------------------------
    # section 12 — prompt builder
    # -----------------------------------------------------------------------
    prompt = build_prompt(query, retrieved_results)

    # -----------------------------------------------------------------------
    # section 13 — generator (Groq)
    # -----------------------------------------------------------------------
    print('Generating response...')
    groq_client      = Groq(api_key=GROQ_API_KEY)
    generated_output = generate(prompt, groq_client)

    # -----------------------------------------------------------------------
    # section 14 — final retrieval report
    # -----------------------------------------------------------------------
    print_report(query, retrieved_results, generated_output, index_path=INDEX_SAVE_PATH)

    # -----------------------------------------------------------------------
    # evaluation — Precision@K, Recall@K, MRR, latency
    # -----------------------------------------------------------------------
    run_evaluation(
        retrieved_results=retrieved_results,
        query=query,
        latency_ms=retrieval_timer.elapsed_ms,
        k=TOP_K,
    )


if __name__ == '__main__':
    run(QUERY)
