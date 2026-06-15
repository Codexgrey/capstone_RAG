"""
generation/llm_client.py — LLM Answer Generation

Routes to the configured LLM backend via LLM_BACKEND in .env.

Supported backends:
    groq        — Groq API (llama-3.1-8b-instant). Fast, free tier available.
    openai      — OpenAI API (gpt-4o-mini).
    anthropic   — Anthropic Claude API (claude-sonnet-4-20250514).
    placeholder — Returns a canned response (no API key needed, for testing).

Set LLM_BACKEND and the corresponding API key in backend/.env.
"""

from typing import List, Dict
from app.config.settings import settings


def generate_answer(
    prompt: str,
    messages: List[Dict[str, str]] = None,
    temperature: float = 0.7,
) -> str:
    """
    Generate an answer from the configured LLM backend.
    Uses chat-message format when available, plain prompt as fallback.

    Args:
        prompt:      Plain-text fallback prompt (used if messages is None).
        messages:    Chat-format message list (preferred).
        temperature: Sampling temperature. Default 0.1 for deterministic RAG
                     responses. Passed through to whichever backend is active.
    """
    backend = settings.LLM_BACKEND

    if backend == "groq":
        return _call_groq(messages or _to_messages(prompt), temperature)

    elif backend == "openai":
        return _call_openai(messages or _to_messages(prompt), temperature)

    elif backend == "anthropic":
        return _call_anthropic(messages or _to_messages(prompt), temperature)

    elif backend == "placeholder":
        return _placeholder(prompt)

    else:
        print(f"⚠️  Unknown LLM_BACKEND '{backend}' — using placeholder")
        return _placeholder(prompt)


# ── Backends ──────────────────────────────────────────────────────────────────

def _call_groq(messages: List[Dict[str, str]], temperature: float = 0.7) -> str:
    """Groq API — llama-3.1-8b-instant. Fast, free tier available."""
    try:
        from groq import Groq
        import os

        api_key = settings.GROQ_API_KEY or os.getenv("GROQ_API_KEY", "")
        client  = Groq(api_key=api_key)
        resp    = client.chat.completions.create(
            model       = "llama-3.1-8b-instant",
            messages    = messages,
            max_tokens  = 1200,
            temperature = temperature,
        )
        return resp.choices[0].message.content

    except ImportError:
        return "Groq package not installed. Run: pip install groq"
    except Exception as e:
        return f"Groq API error: {str(e)}"


def _call_openai(messages: List[Dict[str, str]], temperature: float = 0.1) -> str:
    """
    OpenAI API — gpt-4o-mini.
    Setup: OPENAI_API_KEY in .env, LLM_BACKEND=openai, pip install openai
    """
    try:
        from openai import OpenAI
        import os

        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))
        resp   = client.chat.completions.create(
            model       = "gpt-4o-mini",
            messages    = messages,
            max_tokens  = 1024,
            temperature = temperature,
        )
        return resp.choices[0].message.content

    except ImportError:
        return "OpenAI package not installed. Run: pip install openai"
    except Exception as e:
        return f"OpenAI API error: {str(e)}"


def _call_anthropic(messages: List[Dict[str, str]], temperature: float = 0.1) -> str:
    """
    Anthropic Claude API — claude-sonnet-4-20250514.
    Setup: ANTHROPIC_API_KEY in .env, LLM_BACKEND=anthropic, pip install anthropic
    """
    try:
        import anthropic
        import os

        client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY", ""))

        system_msg   = next((m["content"] for m in messages if m["role"] == "system"), "You are a helpful assistant.")
        conversation = [m for m in messages if m["role"] != "system"]

        resp = client.messages.create(
            model      = "claude-sonnet-4-20250514",
            max_tokens = 1024,
            system     = system_msg,
            messages   = conversation,
            temperature = temperature,
        )
        return resp.content[0].text

    except ImportError:
        return "Anthropic package not installed. Run: pip install anthropic"
    except Exception as e:
        return f"Anthropic API error: {str(e)}"


def _placeholder(prompt: str) -> str:
    """Canned response — no API key needed. Active when LLM_BACKEND=placeholder."""
    return (
        "This is a placeholder response. "
        "Set LLM_BACKEND=groq (or openai/anthropic) in backend/.env "
        "and add the corresponding API key to activate a real model. "
        f"({len(prompt.split())} words of context were prepared.)"
    )


# ── Helper ────────────────────────────────────────────────────────────────────

def _to_messages(prompt: str) -> List[Dict[str, str]]:
    """Wrap a plain prompt string in chat-message format."""
    return [{"role": "user", "content": prompt}]
