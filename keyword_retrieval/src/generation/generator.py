"""
generation/generator.py
========================
LLM answer generation using the Groq API.


What this module does
---------------------
Takes the structured prompt assembled  and sends it to the
Groq LLM (llama-3.1-8b-instant). Returns the generated answer text.

The LLM is instructed (via the prompt) to:
  - Answer ONLY from the retrieved context
  - Follow a strict Answer / Evidence / Citations format
  - Respond with "I don't have enough information" when the context
    is insufficient — never invent facts

Why Groq?
---------
Groq runs llama-3.1-8b-instant with very low latency, making it
practical for interactive RAG pipelines where response time matters.


"""

from config import GROQ_API_KEY, GENERATOR_MODEL, MAX_NEW_TOKENS, TEMPERATURE


# =============================================================================
# STEP 10 — GENERATE ANSWER
# =============================================================================

def generate_answer(prompt: str) -> str:
    """
    Send the RAG prompt to the Groq LLM and return the generated answer.

    How it works:
        1. Initialise the Groq client using the API key from config.py.
        2. Send the prompt as a user message to the LLM.
        3. Return the model's response text.

    The prompt already contains all instructions, context, and formatting
    requirements — this function only handles the API call.

    Parameters
    ----------
    prompt : str
        The structured prompt built by build_prompt() in Step 9.

    Returns
    -------
    str
        The raw answer text from the LLM.
        Sections: Answer / Evidence Used / Citations.

    Raises
    ------
    ValueError    — prompt is empty
    ImportError   — groq library not installed
    RuntimeError  — API call failed or empty response returned
    """
    # --- Validate ---
    if not prompt or not prompt.strip():
        raise ValueError("Prompt cannot be empty.")

    # --- Import groq ---
    try:
        from groq import Groq
    except ImportError:
        raise ImportError(
            "The groq library is required.\n"
            "Run:  pip install groq"
        )

    # --- Validate API key ---
    if not GROQ_API_KEY:
        raise RuntimeError(
            "No Groq API key found.\n"
            "Set GROQ_API_KEY in config.py."
        )

    # --- Send prompt to Groq ---
    try:
        client = Groq(api_key=GROQ_API_KEY)

        response = client.chat.completions.create(
            model       = GENERATOR_MODEL,
            messages    = [{"role": "user", "content": prompt}],
            max_tokens  = MAX_NEW_TOKENS,
            temperature = TEMPERATURE,
        )

        answer = response.choices[0].message.content

        if not answer or not answer.strip():
            raise RuntimeError("LLM returned an empty response.")

        return answer

    except Exception as e:
        raise RuntimeError(
            f"Groq API call failed: {e}\n"
            "Check your API key and internet connection."
        ) from e
