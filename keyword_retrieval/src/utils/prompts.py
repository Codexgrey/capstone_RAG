
# step 9 — Build Prompt


def build_prompt(query: str, retrieved_results: list[dict]) -> str:
   
    # --- Validate inputs ---
    if not query or not query.strip():
        raise ValueError("Query cannot be empty.")

    if not retrieved_results:
        raise ValueError(
            "retrieved_results is empty. "
            "Run Step 8 (Retrieve) before building the prompt."
        )

    # --- Build the context block ---
    # Each retrieved chunk becomes a clearly labelled section
    # so the LLM can reference it precisely.
    context_sections = []

    for item in retrieved_results:
        section = (
            f"{'─' * 70}\n"
            f"Result Rank    : {item['rank']}\n"
            f"Document Title : {item['document_title']}\n"
            f"Source         : {item['source']}\n"
            f"Chunk ID       : {item['chunk_id']}\n"
            f"BM25 Score     : {item['bm25_score']:.4f}\n"
            f"Matched Terms  : {', '.join(item['matched_terms']) if item['matched_terms'] else 'none'}\n"
            f"Citation       : {item['citation']}\n"
            f"\n"
            f"Chunk Text:\n{item['text']}\n"
        )
        context_sections.append(section)

    context_block = "\n".join(context_sections)

    # --- Assemble the full prompt ---
    prompt = f"""\
You are a Keyword-Based Retrieval-Augmented Generation assistant.

YOUR RULES:
1. Answer the question using ONLY the retrieved context provided below.
2. Do NOT invent facts, add outside knowledge, or make assumptions.
3. If the answer cannot be found in the context, respond with exactly:
   "I don't have enough information to answer this question."
4. Always support your answer with evidence from the retrieved chunks.
5. Always list the citations at the end.

REQUIRED OUTPUT FORMAT — follow this exactly:

Answer:
<write a clear 2–4 sentence answer based only on the context below>

Evidence Used:
- <chunk_id> : <one sentence explaining what this chunk contributed>
- <chunk_id> : <one sentence explaining what this chunk contributed>
- <chunk_id> : <one sentence explaining what this chunk contributed>

Citations:
<comma-separated list of citation strings>

─────────────────────────────────────────────────────────────────────────
QUESTION: {query}
─────────────────────────────────────────────────────────────────────────

RETRIEVED CONTEXT:

{context_block}
─────────────────────────────────────────────────────────────────────────

Now write your answer following the required format above.
"""

    return prompt