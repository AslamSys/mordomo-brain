"""
NATS handler for mordomo.brain.generate (request/reply).
"""
import json
import logging
import uuid

from nats.aio.msg import Msg

from src.config import SUBJECT_ACTION
from src.context import get_history, append_exchange
from src.llm import classify, generate
from src.rag import search, upsert
from src.actions import extract_actions
from src.tiers import get_tiers

logger = logging.getLogger(__name__)


async def handle_generate(msg: Msg) -> None:
    """
    Subject: mordomo.brain.generate
    Expects: { text, speaker_id, person_id?, confidence? }
    Replies: { response_text, model_used, tokens_used, actions, tier, intents }

    Flow:
      1. Load conversation history + RAG context
      2. Phase 1 — Groq classifies risk tier + intents (semantic, no keywords)
      3. Phase 2 — Chosen tier generates response with function calling
      4. Publish each action to NATS (orchestrator routes to final destinations)
      5. Reply to orchestrator with full response
    """
    nc = msg._client

    try:
        data = json.loads(msg.data.decode())
    except Exception:
        if msg.reply:
            await nc.publish(msg.reply, json.dumps({"error": "invalid_json"}).encode())
        return

    text       = data.get("text", "").strip()
    speaker_id = data.get("speaker_id", "unknown")

    if not text:
        if msg.reply:
            await nc.publish(msg.reply, json.dumps({"error": "empty_text"}).encode())
        return

    # 1. Load conversation history and RAG context in parallel context
    history     = await get_history(speaker_id)
    rag_context = await search(text, speaker_id)

    # 2. Phase 1 — classify tier semantically via Groq (free, ~200ms)
    tiers = await get_tiers()
    classification = await classify(text, list(tiers.keys()))

    # 3. Phase 2 — generate response with function calling on chosen tier
    try:
        llm_resp = await generate(text, history, rag_context, tier=classification.tier)
    except Exception as exc:
        logger.error("LLM generation failed: %s", exc)
        if msg.reply:
            await nc.publish(msg.reply, json.dumps({"error": "llm_failed", "detail": str(exc)}).encode())
        return

    # 4. Extract actions from tool_calls (already structured — no regex)
    clean_text, actions = extract_actions(llm_resp)

    # 5. Persist exchange to Redis context
    await append_exchange(speaker_id, text, clean_text)

    # 6. Store exchange in Qdrant for future RAG
    await upsert(
        text=f"Usuário: {text}\nMordomo: {clean_text}",
        speaker_id=speaker_id,
        point_id=str(uuid.uuid4()),
    )

    # 7. Publish each detected action to NATS
    #    Orchestrator subscribes to mordomo.brain.action.* and routes to final destinations
    for action in actions:
        action_type = action.get("type", "generic")
        subject = f"{SUBJECT_ACTION}.{action_type}"
        payload = {**action, "speaker_id": speaker_id}
        await nc.publish(subject, json.dumps(payload).encode())
        logger.info("Action published: %s → %s", subject, payload)

    # 8. Reply to orchestrator
    if msg.reply:
        reply_payload = {
            "response_text": clean_text,
            "model_used":    llm_resp.model_used,
            "tokens_used":   llm_resp.tokens_used,
            "actions":       actions,
            "tier":          classification.tier,
            "intents":       classification.intents,
        }
        await nc.publish(msg.reply, json.dumps(reply_payload).encode())

    logger.info(
        "Generated [speaker=%s tier=%s model=%s tokens=%d actions=%d intents=%s]",
        speaker_id, classification.tier, llm_resp.model_used,
        llm_resp.tokens_used, len(actions), classification.intents,
    )
