"""
Calls the language model to generate an answer from the prompt.

Current state:
Right now: placeholder mode (just spits out a hardcoded reply)
Groq API: works out of the box (fast + free, no key needed)
I'm using Groq for now but the options for OpenAi and Anthropic are there but you know :)
Transformer places saved for Collins, Olivier, Nathan

How to actually hook up a real LLM:
first set `LLM_BACKEND` in your `.env`
second throw your API key in `.env`
lastly install whatever package that backend needs

Extra tip:
If you don’t bother with keys, you’ll stay stuck in placeholder mode.
Easiest way: Groq no setup pain, just runs + free ;).
"""
from typing import List, Dict, Any, Optional
from app.config.settings import settings


def generate_answer(
    prompt: str,
    messages: List[Dict[str, str]] = None,
) -> str:
    
    # Generate an answer using the configured LLM backend.
    # Reads LLM_BACKEND from settings to decide which model to use.
    backend = settings.LLM_BACKEND

    if backend == "placeholder":
        return _placeholder(prompt)

    elif backend == "groq":
        return _call_groq(messages or _prompt_to_messages(prompt))

    elif backend == "openai":
        return _call_openai(messages or _prompt_to_messages(prompt))

    elif backend == "anthropic":
        return _call_anthropic(messages or _prompt_to_messages(prompt))

    # Transformer slots for my teammates
    elif backend == "transformer_a":
        # Collins (vector retrieval model)
        return _call_transformer_a(prompt)

    elif backend == "transformer_b":
        # Olivier (keyword retrieval model)
        return _call_transformer_b(prompt)

    elif backend == "transformer_c":
        # Nathan (CLaRA retrieval model)
        return _call_transformer_c(prompt)

    else:
        # Default placeholder until a backend is configured
        print(f"⚠️ Unknown LLM_BACKEND '{backend}' — using placeholder")
        return _placeholder(prompt)


# Placeholder
def _placeholder(prompt: str) -> str:
    """
    Returns a hardcoded response.
    Used when no LLM is configured yet.
    Active when LLM_BACKEND=local_llm or LLM_BACKEND=placeholder
    """
    return (
        "This is a placeholder response. "
        "No LLM backend is configured yet. "
        "Set LLM_BACKEND in your .env to activate a real model. "
        f"Your question was received and {len(prompt.split())} words of context were prepared."
    )

# For now I only want to test with Groq.
# OpenAi/Anthropic code is parked here for later if it's needed
# Groq API
def _call_groq(messages: List[Dict[str, str]]) -> str:
    try:
        from groq import Groq
        import os

        client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=messages,
            max_tokens=1024,
            temperature=0.1,
        )
        return response.choices[0].message.content

    except ImportError:
        return "Groq package not installed. Run: pip install groq"
    except Exception as e:
        return f"Groq API error: {str(e)}"


# OpenAI
# def _call_openai(messages: List[Dict[str, str]]) -> str:
#     """
#     Call OpenAI API.
# 
#     Setup:
#         1. Get API key at: https://platform.openai.com
#         2. Add to .env: OPENAI_API_KEY=your_key_here
#         3. Add to .env: LLM_BACKEND=openai
#         4. pip install openai
#     """
#     try:
#         from openai import OpenAI
#         import os
# 
#         client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
#         response = client.chat.completions.create(
#             model="gpt-4o-mini",
#             messages=messages,
#             max_tokens=1024,
#             temperature=0.1,
#         )
#         return response.choices[0].message.content
# 
#     except ImportError:
#         return "OpenAI package not installed. Run: pip install openai"
#     except Exception as e:
#         return f"OpenAI API error: {str(e)}"
# 
# 
# # Anthropic (Claude) 
# def _call_anthropic(messages: List[Dict[str, str]]) -> str:
#     """
#     Call Anthropic Claude API.
# 
#     Setup:
#         1. Get API key at: https://console.anthropic.com
#         2. Add to .env: ANTHROPIC_API_KEY=your_key_here
#         3. Add to .env: LLM_BACKEND=anthropic
#         4. pip install anthropic
#     """
#     try:
#         import anthropic
#         import os
# 
#         client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
# 
#         # Separate system message from conversation
#         system_msg = next(
#             (m["content"] for m in messages if m["role"] == "system"),
#             "You are a helpful assistant."
#         )
#         conversation = [m for m in messages if m["role"] != "system"]
# 
#         response = client.messages.create(
#             model="claude-sonnet-4-20250514",
#             max_tokens=1024,
#             system=system_msg,
#             messages=conversation,
#         )
#         return response.content[0].text
# 
#     except ImportError:
#         return "Anthropic package not installed. Run: pip install anthropic"
#     except Exception as e:
#         return f"Anthropic API error: {str(e)}"
# 

#Transformer slots 
def _call_transformer_a(prompt: str) -> str:
    # Collins's vector retrieval model.
    # To be implemented when vector/ module is ready.
    # TODO: Collins  implement your model here
    return "transformer_a not implemented yet, Collins's model slot"

def _call_transformer_b(prompt: str) -> str:
    # Olivier's keyword retrieval model.
    # To be implemented when keyword/ module is ready.
    # TODO: Olivier implement your model here
    return "transformer_b not implemented yet, Olivier's model slot"

def _call_transformer_c(prompt: str) -> str:
    # Nathan's CLaRA retrieval model.
    # To be implemented when clara/ module is ready.
    # TODO: Nathan implement your model here
    return "🔒 transformer_c not implemented yet, Nathan's model slot"

# Helper 
def _prompt_to_messages(prompt: str) -> List[Dict[str, str]]:
    # Convert a plain prompt string to chat message format.
    return [{"role": "user", "content": prompt}]