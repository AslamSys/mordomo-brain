"""
LLM client — calls the LiteLLM gateway (already running in infra on port 4000).
The gateway holds all API keys and exposes virtual model names.
"""
import logging
from dataclasses import dataclass

import httpx

from src.config import LITELLM_URL, MODEL_SIMPLE, MODEL_COMPLEX, MODEL_STAKES, SYSTEM_PROMPT

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

    payload = {
        "model":       model,
        "messages":    messages,
        "max_tokens":  512,
        "temperature": 0.7,
    }

    # Try primary model, fallback to MODEL_SIMPLE
    for attempt_model in [model, MODEL_SIMPLE]:
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.post(
                    f"{LITELLM_URL}/chat/completions",
                    json={**payload, "model": attempt_model},
                )
                resp.raise_for_status()
                data = resp.json()
                content = data["choices"][0]["message"]["content"]
                tokens  = data.get("usage", {}).get("total_tokens", 0)
                return LLMResponse(text=content, model_used=attempt_model, tokens_used=tokens)
        except Exception as exc:
            logger.warning("LLM call failed [model=%s]: %s", attempt_model, exc)
            if attempt_model == MODEL_SIMPLE:
                raise

    raise RuntimeError("All LLM attempts failed")
