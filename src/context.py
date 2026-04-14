"""
Redis-backed conversation context per speaker_id.
Stores the last N messages as a JSON list in a Redis key with TTL.
Uses Redis db1 (mordomo-general).
"""
import json
import logging
from typing import Optional

import redis.asyncio as aioredis

from src.config import REDIS_URL, CONTEXT_MAX_MESSAGES, CONTEXT_TTL_SECONDS

logger = logging.getLogger(__name__)

_redis: Optional[aioredis.Redis] = None


async def get_redis() -> aioredis.Redis:
    global _redis
    if _redis is None:
        _redis = await aioredis.from_url(REDIS_URL, decode_responses=True)
    return _redis


def _key(speaker_id: str) -> str:
    return f"brain:context:{speaker_id}"


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
    """Append a user/assistant exchange and trim to CONTEXT_MAX_MESSAGES pairs."""
    try:
        r = await get_redis()
        history = await get_history(speaker_id)
        history.append({"role": "user",      "content": user_text})
        history.append({"role": "assistant", "content": assistant_text})
        # Keep only last N messages (pairs)
        if len(history) > CONTEXT_MAX_MESSAGES:
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
