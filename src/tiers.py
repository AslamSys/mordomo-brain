"""
Dynamic tier discovery from LiteLLM gateway /models endpoint.

The brain calls GET /models on startup and every TIER_CACHE_TTL seconds.
This means: change a model_name in litellm_config.yaml + SIGHUP the gateway
→ brain picks it up automatically within TTL, no code change needed.

Tiers are filtered by TIER_PREFIX (default "mordomo-") and exclude
embeddings models (which are not conversational tiers).
"""
import logging
import time

import httpx

from src.config import LITELLM_URL, LITELLM_MASTER_KEY, TIER_PREFIX, TIER_CACHE_TTL, TIER_FALLBACK

logger = logging.getLogger(__name__)

# In-memory cache
_cached_tiers: list[str] = []
_cache_ts: float = 0.0

# Models that start with TIER_PREFIX but are NOT conversational tiers
_EXCLUDED_SUFFIXES = ("embeddings", "embedding")


async def fetch_tiers() -> list[str]:
    """
    Fetch available tiers from LiteLLM gateway.
    Filters by TIER_PREFIX and excludes embedding models.
    Returns empty list on failure — caller must handle gracefully.
    """
    headers = {}
    if LITELLM_MASTER_KEY:
        headers["Authorization"] = f"Bearer {LITELLM_MASTER_KEY}"

    try:
        async with httpx.AsyncClient(timeout=5) as client:
            resp = await client.get(f"{LITELLM_URL}/models", headers=headers)
            resp.raise_for_status()
            data = resp.json()

        model_ids: list[str] = [m["id"] for m in data.get("data", [])]
        tiers = sorted(
            m for m in model_ids
            if m.startswith(TIER_PREFIX)
            and not any(m.endswith(s) for s in _EXCLUDED_SUFFIXES)
        )
        logger.info("Tiers discovered from gateway (%d): %s", len(tiers), tiers)
        return tiers

    except Exception as exc:
        logger.warning("Failed to fetch tiers from gateway: %s", exc)
        return []


async def get_tiers() -> list[str]:
    """
    Return cached tiers, refreshing if TTL expired.
    If gateway is unreachable and no cache exists, returns [TIER_FALLBACK]
    so the brain can still operate in degraded mode.
    """
    global _cached_tiers, _cache_ts

    now = time.monotonic()
    if not _cached_tiers or (now - _cache_ts) > TIER_CACHE_TTL:
        fresh = await fetch_tiers()
        if fresh:
            _cached_tiers = fresh
            _cache_ts = now
        elif not _cached_tiers:
            logger.error(
                "No tiers available and no cache — gateway may be down. "
                "Falling back to: %s", TIER_FALLBACK
            )
            _cached_tiers = [TIER_FALLBACK]
            _cache_ts = now

    return _cached_tiers


async def init_tiers() -> None:
    """
    Called on startup to warm the cache before accepting any NATS messages.
    Logs a warning (not error) if gateway is not yet ready — brain will retry
    on first request via get_tiers().
    """
    tiers = await get_tiers()
    if tiers == [TIER_FALLBACK]:
        logger.warning(
            "Brain started with fallback tier only (%s). "
            "Will retry tier discovery on next request.", TIER_FALLBACK
        )
