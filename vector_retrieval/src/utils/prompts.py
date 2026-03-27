"""
src/utils/prompts.py
Helper utility — builds the structured RAG prompt from a query and retrieved chunks.
"""

from typing import List, Dict, Any


def build_prompt(query: str, retrieved_results: List[Dict[str, Any]]) -> str:
    """
    Construct a structured prompt for the generator from the user query and
    the top retrieved chunks.

    The prompt instructs the model to return its response in exactly three
    labelled sections: Answer, Evidence Used, and Citations.

    Args:
        query:             The user's natural language question.
        retrieved_results: List of result dicts from the retriever, each
                           containing rank, document_title, source, chunk_id,
                           similarity, citation, and text.

    Returns:
        Formatted prompt string ready to be passed to the generator.
    """
    context_sections = []

    for item in retrieved_results:
        section = (
            f"\n{'-' * 100}\n"
            f"Result Rank: {item['rank']}\n"
            f"Document Title: {item['document_title']}\n"
            f"Source: {item['source']}\n"
            f"Chunk ID: {item['chunk_id']}\n"
            f"Similarity: {item['similarity']:.4f}\n"
            f"Citation: {item['citation']}\n"
            f"\nChunk Text: {item['text']}\n"
        )
        context_sections.append(section)

    context_block = '\n\n'.join(context_sections)

    prompt = f"""
    You are assisting with a Retrieval-Augmented Generation System.
    Use only the retrieved context below. Do not invent facts.

    Return your answer in exactly this format:
    Answer:
    <2-4 sentence answer to the question>
    <space before producing "Evidence Used" for visual hierarchy>

    Evidence Used:
    bullet list with proper spacing between bullets for good visibility
    - <bullet with chunk id and what it contributed>
    - <bullet with chunk id and what it contributed>
    - <bullet with chunk id and what it contributed>
    <space before producing "Citations" for visual hierarchy>

    Citations:
    <comma-separated citations>

    Question: {query}

    Retrieved Context:
    {context_block}

    Now produce the final response.
    """

    return prompt
