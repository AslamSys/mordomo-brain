"""
Redis-backed conversation context per speaker_id.
Stores the last N messages as a JSON list in a Redis key with TTL.
Uses Redis db1 (mordomo-general).

When history exceeds CONTEXT_MAX_MESSAGES, the older portion is summarized via
LLM before being discarded (CONTEXT_SUMMARIZE_KEEP recent messages preserved verbatim).
"""
import json
import logging
from typing import Optional

import httpx
import redis.asyncio as aioredis

from src.config import (
    REDIS_URL, CONTEXT_MAX_MESSAGES, CONTEXT_TTL_SECONDS,
    CONTEXT_SUMMARIZE_KEEP, LITELLM_URL, TIER_FALLBACK,
    GROQ_API_KEY, GROQ_URL, GROQ_MODEL,
)

logger = logging.getLogger(__name__)

_redis: Optional[aioredis.Redis] = None


async def get_redis() -> aioredis.Redis:
    global _redis
    if _redis is None:
        _redis = await aioredis.from_url(REDIS_URL, decode_responses=True)
    return _redis


def _key(speaker_id: str) -> str:
    return f"brain:context:{speaker_id}"


async def _summarize_messages(messages: list[dict]) -> str:
    """
    Ask the LLM to summarize a list of messages into a compact paragraph.
    Tries LiteLLM gateway first, then Groq direct.
    """
    system = (
        "Você é um assistente de memória. Resuma a conversa abaixo em até 5 frases "
        "em português, preservando apenas os fatos e decisões mais importantes."
    )
    conversation_text = "\n".join(
        f"{m['role'].upper()}: {m['content']}" for m in messages
        if m.get("role") in ("user", "assistant")
    )
    payload_messages = [
        {"role": "system",  "content": system},
        {"role": "user",    "content": conversation_text},
    ]
    payload = {"model": TIER_FALLBACK, "messages": payload_messages, "max_tokens": 200, "temperature": 0.3}

    # Try LiteLLM gateway
    try:
        async with httpx.AsyncClient(timeout=20) as client:
            resp = await client.post(f"{LITELLM_URL}/chat/completions", json=payload)
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
    except Exception as exc:
        logger.warning("LiteLLM summarization failed: %s — trying Groq direct", exc)

    # Fallback: Groq direct
    if GROQ_API_KEY:
        try:
            async with httpx.AsyncClient(timeout=20) as client:
                resp = await client.post(
                    f"{GROQ_URL}/chat/completions",
                    headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
                    json={**payload, "model": GROQ_MODEL},
                )
                resp.raise_for_status()
                return resp.json()["choices"][0]["message"]["content"]
        except Exception as exc:
            logger.warning("Groq summarization failed: %s", exc)

    raise RuntimeError("All summarization attempts failed")


async def get_history(speaker_id: str) -> list[dict]:
    """Return the conversation history for a speaker as a list of {role, content} dicts."""
    try:
        r = await get_redis()
        raw = await r.get(_key(speaker_id))
        if raw:
            return json.loads(raw)
    except Exception as e:
        logger.warning("Failed to get context from Redis: %s", e)
    return []


async def append_exchange(speaker_id: str, user_text: str, assistant_text: str) -> None:
    """
    Append a user/assistant exchange.
    If history exceeds CONTEXT_MAX_MESSAGES, summarize the older portion first.
    """
    try:
        r = await get_redis()
        history = await get_history(speaker_id)
        history.append({"role": "user",      "content": user_text})
        history.append({"role": "assistant", "content": assistant_text})

        if len(history) > CONTEXT_MAX_MESSAGES:
            keep = CONTEXT_SUMMARIZE_KEEP
            to_summarize = history[:-keep]
            recent       = history[-keep:]
            try:
                summary = await _summarize_messages(to_summarize)
                history = [{"role": "system", "content": f"[Resumo do histórico anterior]: {summary}"}] + recent
                logger.info("Context summarized for speaker=%s (%d→%d msgs)", speaker_id, len(to_summarize) + keep, len(history))
            except Exception as exc:
                logger.warning("Summarization unavailable, truncating instead: %s", exc)
                history = history[-CONTEXT_MAX_MESSAGES:]

        await r.set(_key(speaker_id), json.dumps(history), ex=CONTEXT_TTL_SECONDS)
    except Exception as e:
        logger.warning("Failed to save context to Redis: %s", e)


async def clear_context(speaker_id: str) -> None:
    try:
        r = await get_redis()
        await r.delete(_key(speaker_id))
    except Exception as e:
        logger.warning("Failed to clear context: %s", e)
