# takes raw LLM answer + retrieved chunks
# builds the team’s shared output format:
# "answer": final text from the model
# "citations": list of sources/pages used
# "retrieval_method": how chunks were pulled (e.g. vector/keyword)
# "latency_ms": time taken for retrieval + generation
# makes sure everything matches the answer_response schema exactly
from typing import List, Dict, Any, Optional


def format_response(
    answer: str,
    chunks: List[Dict[str, Any]],
    retrieval_method: str = "none",
    latency_ms: float = 0.0,
    session_id: str = None,
) -> Dict[str, Any]:
    # Format the final response matching the team's shared contract.
    # "answer": model reply
    # "citations": sources/pages
    # "retrieval_method": vector/keyword/clara/none
    # "latency_ms": time in ms
    # "session_id": chat UUID
    citations = _build_citations(chunks)

    response = {
        "answer" : answer,
        "citations" : citations,
        "retrieval_method" : retrieval_method,
        "latency_ms" : round(latency_ms, 2),
    }

    if session_id:
        response["session_id"] = session_id

    return response


def _build_citations(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    # Build citation objects from retrieved chunks.
    # going to have chunk_id,, source_name, page, section (if available).
    # Deduplicates citations by source_name + page so the same
    # page is not cited twice even if multiple chunks came from it.
    citations = []
    seen = set()

    for chunk in chunks:
        metadata = chunk.get("metadata", {})
        source_name = chunk.get("source_name", "Unknown")
        page = metadata.get("page", None)
        chunk_id = chunk.get("chunk_id", "")
        section = metadata.get("section", None)

        # Deduplicate by source + page
        key = f"{source_name}_{page}"
        if key in seen:
            continue
        seen.add(key)

        citations.append({
            "chunk_id" : chunk_id,
            "source_name" : source_name,
            "page" : page,
            "section" : section,
        })

    return citations


def format_error_response(
    error_message: str,
    retrieval_method: str = "none",
) -> Dict[str, Any]:
    # Format an error response when something goes wrong.
    # Returns a safe response the frontend can display.
    return {
        "answer" : f"An error occurred: {error_message}",
        "citations" : [],
        "retrieval_method" : retrieval_method,
        "latency_ms" : 0.0,
    }