"""
Prompt Construction
Builds the prompt that gets sent to the LLM.

Takes retrieved chunks from ChromaDB and the user's question,
and constructs a grounded prompt that tells the LLM to answer
only based on the provided context at the end sends it to the llm.

"""
from typing import List, Dict, Any


def build_prompt(question: str, chunks: List[Dict[str, Any]]) -> str:
    """
    Build a grounded prompt from the user's question and retrieved chunks.

    Args:
        question: The user's question
        chunks: List of retrieved chunk dicts from indexer.search_chunks()
        Each chunk has: text, source_name, page, score
    Returns: Full prompt string ready to send to the LLM
    """
    # if no chunks found, fall back to a simple prompt
    if not chunks:
        return _build_no_context_prompt(question)

    # Build context block from chunks
    context_parts = []
    for i, chunk in enumerate(chunks, start=1):
        # here is to extract the source name and page for citation
        source = chunk.get("source_name", "Unknown")
        page = chunk.get("metadata", {}).get("page", "N/A")
        text = chunk.get("text", "")

        # format chunk into readable block
        context_parts.append(
            f"[Source {i}: {source}, Page {page}]\n{text}"
        )

    # Join all chunks into one context string
    context = "\n\n".join(context_parts)

    # final prompt that instructs llm to use context
    prompt = f"""You are a helpful assistant. Answer the question below using ONLY the provided context.
If the answer is not in the context, say "I could not find an answer in the provided documents."
Do not make up information. Keep your answer concise and accurate.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:"""

    return prompt


def build_chat_prompt(
    question : str,
    chunks : List[Dict[str, Any]],
    chat_history : List[Dict[str, str]] = None
) -> List[Dict[str, str]]:
    """
    Build a chat-style prompt with message history.
    Used for API-based models for now just Groq but we can add (OpenAI, Anthropic, Claude).

    Args:
        question: The user's question
        chunks: Retrieved chunks from ChromaDB
        chat_history: Previous messages [{"role": "user/assistant", "content": "..."}]

    Returns:
        List of message dicts in chat format
    """
    messages = []

    # System message
    messages.append({
        "role": "system",
        "content": (
            "You are a helpful assistant for a RAG system. "
            "Answer questions using ONLY the provided context. "
            "If the answer is not in the context, say so clearly. "
            "Always cite the source document and page number when possible."
        )
    })

    # Add chat history if provided
    if chat_history:
        for msg in chat_history:
            messages.append({
                "role" : msg["role"],
                "content" : msg["content"]
            })

    # Build context from chunks
    if chunks:
        context_parts = []
        for i, chunk in enumerate(chunks, start=1):
            source = chunk.get("source_name", "Unknown")
            page = chunk.get("metadata", {}).get("page", "N/A")
            text = chunk.get("text", "")
            context_parts.append(f"[Source {i}: {source}, Page {page}]\n{text}")

        context = "\n\n".join(context_parts)
        user_content = f"Context:\n{context}\n\nQuestion: {question}"
    else:
        user_content = (
            f"No relevant context was found in the documents.\n\nQuestion: {question}"
        )

    messages.append({
        "role" : "user",
        "content" : user_content
    })

    return messages


def _build_no_context_prompt(question: str) -> str:
    """Fallback prompt when no chunks are retrieved."""
    return f"""You are a helpful assistant.
No relevant context was found in the uploaded documents for this question.
Please let the user know and suggest they upload relevant documents.

QUESTION:
{question}

ANSWER:"""