"""Tier discovery via Redis db1 (mordomo:tiers / mordomo:tiers:fallbacks)."""
import logging
import time

import redis.asyncio as aioredis

from src.config import REDIS_URL, TIER_CACHE_TTL, TIER_FALLBACK, TIER_STRICT_MODE

logger = logging.getLogger(__name__)

REDIS_KEY_MODELS    = "mordomo:tiers"
REDIS_KEY_FALLBACKS = "mordomo:tiers:fallbacks"

_redis: aioredis.Redis | None = None

# In-memory cache
_cached_tiers: dict[str, dict[str, str]] = {}
_cache_ts: float = 0.0


def _get_redis() -> aioredis.Redis:
    global _redis
    if _redis is None:
        _redis = aioredis.from_url(REDIS_URL, decode_responses=True)
    return _redis


async def fetch_tiers() -> dict[str, dict[str, str]]:
    """
    Lê mordomo:tiers e mordomo:tiers:fallbacks do Redis.
    Retorna { tier_name: {"model": "provider/model", "fallback": "provider/model"} }.
    """
    r = _get_redis()
    try:
        models    = await r.hgetall(REDIS_KEY_MODELS)
        fallbacks = await r.hgetall(REDIS_KEY_FALLBACKS)

        return {
            tier: {
                "model":    model,
                "fallback": fallbacks.get(tier, ""),
            }
            for tier, model in models.items()
        }
    except Exception as exc:
        logger.warning("Failed to fetch tiers from Redis: %s", exc)
        return {}


async def get_tiers() -> dict[str, dict[str, str]]:
    """
    Retorna tiers cacheados, atualizando se TTL expirou.
    Se Redis estiver inacessível, mantém o último cache válido.
    """
    global _cached_tiers, _cache_ts

    now = time.monotonic()
    if not _cached_tiers or (now - _cache_ts) > TIER_CACHE_TTL:
        fresh = await fetch_tiers()
        if fresh:
            _cached_tiers = fresh
            _cache_ts = now
        elif not _cached_tiers:
            logger.error("No tiers available and no cache — Redis may be down")

    return _cached_tiers


async def resolve_tier(tier_name: str) -> tuple[str, str]:
    """
    Retorna (model, fallback_model) para um tier semântico.
    Se o tier não existir, usa TIER_FALLBACK como fallback semântico.
    """
    tiers = await get_tiers()
    entry = tiers.get(tier_name) or tiers.get(TIER_FALLBACK) or {}
    model = entry.get("model", "")
    fallback = entry.get("fallback", "")
    return model, fallback


async def init_tiers() -> None:
    """Warm tier cache and optionally enforce strict startup policy."""
    tiers = await get_tiers()
    if tiers:
        logger.info("Tiers loaded from Redis (%d): %s", len(tiers), list(tiers.keys()))
        return

    msg = (
        "No tiers found in Redis (db1). Expected keys: "
        f"'{REDIS_KEY_MODELS}' and '{REDIS_KEY_FALLBACKS}'"
    )
    if TIER_STRICT_MODE:
        raise RuntimeError(msg)
    logger.warning("%s — continuing in degraded mode", msg)
