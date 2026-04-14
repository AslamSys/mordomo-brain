"""
RAG via Qdrant — searches for semantically relevant past context.
Optional: if Qdrant is unreachable, gracefully returns empty list.
"""
import logging
from typing import Optional

import httpx

from src.config import QDRANT_URL, QDRANT_COLLECTION, RAG_ENABLED, RAG_TOP_K, RAG_MIN_SCORE, LITELLM_URL

logger = logging.getLogger(__name__)


async def _get_embedding(text: str) -> Optional[list[float]]:
    """Get text embedding via LiteLLM gateway."""
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(
                f"{LITELLM_URL}/embeddings",
                json={"model": "mordomo-embeddings", "input": text},
            )
            resp.raise_for_status()
            return resp.json()["data"][0]["embedding"]
    except Exception as e:
        logger.warning("Embedding request failed: %s", e)
        return None


async def search(text: str, speaker_id: str) -> list[str]:
    """Return list of relevant text snippets from Qdrant, or empty list on failure."""
    if not RAG_ENABLED:
        return []

    embedding = await _get_embedding(text)
    if not embedding:
        return []

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(
                f"{QDRANT_URL}/collections/{QDRANT_COLLECTION}/points/search",
                json={
                    "vector": embedding,
                    "limit": RAG_TOP_K,
                    "score_threshold": RAG_MIN_SCORE,
                    "filter": {
                        "must": [{"key": "speaker_id", "match": {"value": speaker_id}}]
                    },
                    "with_payload": True,
                },
            )
            resp.raise_for_status()
            hits = resp.json().get("result", [])
            return [h["payload"].get("text", "") for h in hits if h.get("payload")]
    except Exception as e:
        logger.warning("Qdrant search failed: %s", e)
        return []


async def upsert(text: str, speaker_id: str, point_id: str) -> None:
    """Store a new text snippet in Qdrant for future RAG retrieval."""
    if not RAG_ENABLED:
        return

    embedding = await _get_embedding(text)
    if not embedding:
        return

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            await client.put(
                f"{QDRANT_URL}/collections/{QDRANT_COLLECTION}/points",
                json={
                    "points": [{
                        "id":      point_id,
                        "vector":  embedding,
                        "payload": {"text": text, "speaker_id": speaker_id},
                    }]
                },
            )
    except Exception as e:
        logger.warning("Qdrant upsert failed: %s", e)
