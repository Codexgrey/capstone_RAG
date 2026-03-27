"""
src/main.py
Local entry point for the Vector Retrieval module.

Run from the vector_retrieval/ root:
    python -m src.main

Pipeline  (mirrors notebook Sections 2 - 12, then Evaluation):
    load_test_document   (Section 2 + 3)
    chunker              (Section 4)
    embedding model      (Section 5 + 6)
    FAISS index          (Section 7)
    retriever            (Section 8)     <- timed for latency measurement
    prompt builder       (Section 10)
    generator (Groq)     (Section 11)
    formatted report     (Section 12)
    evaluation report    (Evaluation)
"""

import os
from src.utils.loader import load_test_document
from src.utils.chunker import chunk_text_with_metadata
from src.models.embedding_model import load_embedding_model, encode_chunks
from src.indexing.vector_store import build_index
from src.retrieval.retriever import retrieve
from src.utils.prompts import build_prompt
from src.utils.response_printer import print_report
from src.evaluation.evaluate import Timer, run_evaluation
from groq import Groq
from dotenv import load_dotenv


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------
CHUNK_SIZE    = 150
CHUNK_OVERLAP = 30
TOP_K         = 3

load_dotenv()
GROQ_API_KEY         = os.environ.get('GROQ_API_KEY', 'YOUR_GROQ_API_KEY_HERE')
GENERATOR_MODEL_NAME = 'llama-3.1-8b-instant'


# sample query to test the full pipeline end-to-end 
QUERY = 'What role do Transformers play in RAG Systems?'


def generate(prompt: str, client: Groq, max_new_tokens: int = 500) -> str:
    """Send a prompt to the Groq LLM and return the generated text."""
    response = client.chat.completions.create(
        model=GENERATOR_MODEL_NAME,
        messages=[{'role': 'user', 'content': prompt}],
        max_tokens=max_new_tokens,
        temperature=0.1
    )
    return response.choices[0].message.content


def run(query: str) -> None:
    """
    Execute the full vector retrieval RAG pipeline for a given query,
    then evaluate retrieval quality and latency.

    Args:
        query: Natural language question to answer from the document.
    """
    # -----------------------------------------------------------------------
    # sections 2, 3 — test document setup and loader
    # -----------------------------------------------------------------------
    document_text, document_title, document_source, document_id = load_test_document()

    # -----------------------------------------------------------------------
    # section 4 — chunker
    # -----------------------------------------------------------------------
    print('Chunking document...')
    chunk_records = chunk_text_with_metadata(
        document_text,
        chunk_size=CHUNK_SIZE,
        overlap=CHUNK_OVERLAP,
        document_title=document_title,
        source=document_source,
        document_id=document_id
    )
    print(f'Number of chunks : {len(chunk_records)}')
    print(f'Chunk size       : {CHUNK_SIZE}')
    print(f'Chunk overlap    : {CHUNK_OVERLAP}')
    print()

    # -----------------------------------------------------------------------
    # sections 5, 6 — embedding model and encoding
    # -----------------------------------------------------------------------
    print('Loading embedding model and encoding chunks...')
    embedding_model = load_embedding_model()
    chunk_texts = [chunk['text'] for chunk in chunk_records]
    embeddings = encode_chunks(embedding_model, chunk_texts)
    print(f'Embeddings shape : {embeddings.shape}')
    print(f'Embedding dtype  : {embeddings.dtype}')
    print()

    # -----------------------------------------------------------------------
    # section 7 — FAISS index
    # -----------------------------------------------------------------------
    print('Building FAISS index...')
    index = build_index(embeddings)
    print(f'FAISS index built. Vectors stored: {index.ntotal}')
    print()

    # -----------------------------------------------------------------------
    # section 8 — retriever  (timed for latency)
    # -----------------------------------------------------------------------
    with Timer() as retrieval_timer:
        retrieved_results = retrieve(
            query, embedding_model, index, chunk_records, top_k=TOP_K
        )

    # -----------------------------------------------------------------------
    # section 10 — prompt builder
    # -----------------------------------------------------------------------
    prompt = build_prompt(query, retrieved_results)

    # -----------------------------------------------------------------------
    # section 11 — generator (Groq)
    # -----------------------------------------------------------------------
    print('Generating response...')
    groq_client = Groq(api_key=GROQ_API_KEY)
    generated_output = generate(prompt, groq_client)

    # -----------------------------------------------------------------------
    # section 12 — final retrieval report
    # -----------------------------------------------------------------------
    print_report(query, retrieved_results, generated_output)

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
