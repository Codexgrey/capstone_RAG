def build_prompt(query, retrieved_results):
    """
    Builds a structured prompt from the query and retrieved chunks.
    Compatible with vector, keyword, and hybrid retrieval results.
    """
    context_sections = []

    for item in retrieved_results:
        # ─── Score line — compatible vector, keyword, hybrid ───────
        if 'rrf_score' in item:
            score_line = f"RRF Score      : {item['rrf_score']:.6f}\n"
            source_line = f"Source         : {item.get('retrieval', 'hybrid')}\n"
        elif 'similarity' in item and item['similarity'] is not None:
            score_line  = f"Similarity     : {item['similarity']:.4f}\n"
            source_line = ''
        elif 'bm25_score' in item and item['bm25_score'] is not None:
            score_line  = f"BM25 Score     : {item['bm25_score']:.4f}\n"
            source_line = ''
        else:
            score_line  = ''
            source_line = ''

        section = (
            f"\n{'—' * 50}\n"
            f"Result Rank    : {item['rank']}\n"
            f"Document Title : {item['document_title']}\n"
            f"Source         : {item['source']}\n"
            f"Chunk ID       : {item['chunk_id']}\n"
            + score_line
            + source_line +
            f"Citation       : {item['citation']}\n"
            f"\nChunk Text : {item['text']}\n"
        )
        context_sections.append(section)

    context_block = '\n\n'.join(context_sections)

    prompt = f"""You are assisting with a Retrieval-Augmented Generation System.
Use only the retrieved context below. Do not invent facts.
If the answer is not present in the context, respond with exactly: "I don't have enough information."

Return your answer in exactly this format:

Answer:
<2-4 sentence answer to the question>

Evidence Used:
- <chunk_id> : <what it contributed>
- <chunk_id> : <what it contributed>
- <chunk_id> : <what it contributed>

Citations:
<comma-separated citations>

Question: {query}

Retrieved Context:
{context_block}

Now produce the final response.
"""
    return prompt