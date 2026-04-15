"""
LLM client — calls the LiteLLM gateway (already running in infra on port 4000).
The gateway holds all API keys and exposes virtual model names.

Fallback chain:
  1. LiteLLM gateway (handles provider rotation + its own fallbacks internally)
  2. Groq direct (bypasses gateway — funciona com apenas GROQ_API_KEY, mesmo sem gateway)
"""
import logging
from dataclasses import dataclass

import httpx

from src.config import (
    LITELLM_URL, MODEL_SIMPLE, MODEL_COMPLEX, MODEL_STAKES,
    GROQ_API_KEY, GROQ_URL, GROQ_MODEL, SYSTEM_PROMPT,
)

logger = logging.getLogger(__name__)

_HIGH_STAKES_KEYWORDS = ("pix", "transfere", "pagar", "senha", "api key", "alarme", "emergência")
_COMPLEX_KEYWORDS = ("explique", "analise", "compare", "por que", "como funciona", "resumo", "calcule")


def _select_model(text: str) -> str:
    lower = text.lower()
    if any(k in lower for k in _HIGH_STAKES_KEYWORDS):
        return MODEL_STAKES
    if any(k in lower for k in _COMPLEX_KEYWORDS) or len(text) > 200:
        return MODEL_COMPLEX
    return MODEL_SIMPLE


@dataclass
class LLMResponse:
    text: str
    model_used: str
    tokens_used: int


async def _call_litellm(model: str, messages: list[dict]) -> LLMResponse:
    """Call LiteLLM gateway. Raises on failure."""
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            f"{LITELLM_URL}/chat/completions",
            json={"model": model, "messages": messages, "max_tokens": 512, "temperature": 0.7},
        )
        resp.raise_for_status()
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        tokens  = data.get("usage", {}).get("total_tokens", 0)
        return LLMResponse(text=content, model_used=model, tokens_used=tokens)


async def _call_groq_direct(messages: list[dict]) -> LLMResponse:
    """
    Fallback direto para Groq — não depende do LiteLLM gateway.
    Usado quando: gateway unavailable, nenhuma API key cloud configurada,
    ou todas as tentativas anteriores falharam.
    """
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY não configurado — sem fallback disponível")

    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            f"{GROQ_URL}/chat/completions",
            headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
            json={"model": GROQ_MODEL, "messages": messages, "max_tokens": 512, "temperature": 0.7},
        )
        resp.raise_for_status()
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        tokens  = data.get("usage", {}).get("total_tokens", 0)
        return LLMResponse(text=content, model_used=f"groq/{GROQ_MODEL}", tokens_used=tokens)


async def generate(
    user_text: str,
    history: list[dict],
    rag_context: list[str],
    speaker_id: str,
    force_model: str | None = None,
) -> LLMResponse:
    model = force_model or _select_model(user_text)

    # Build messages
    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]

    if rag_context:
        context_block = "\n\n".join(f"- {c}" for c in rag_context)
        messages.append({
            "role": "system",
            "content": f"Contexto relevante recuperado:\n{context_block}"
        })

    messages.extend(history)
    messages.append({"role": "user", "content": user_text})

    # 1. Try LiteLLM gateway (handles its own provider fallbacks internally)
    try:
        return await _call_litellm(model, messages)
    except Exception as exc:
        logger.warning("LiteLLM gateway failed [model=%s]: %s — trying Groq direct", model, exc)

    # 2. Groq direct — último recurso, independente do gateway
    return await _call_groq_direct(messages)
