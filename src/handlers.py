"""
NATS handler for mordomo.brain.generate (request/reply).
Streams response sentences to TTS as they arrive from the LLM.
"""
import json
import logging
import time
import uuid

from nats.aio.msg import Msg

from src.config import SUBJECT_ACTION
from src.context import get_history, append_exchange
from src.llm import classify, generate_stream
from src.rag import search, upsert
from src.actions import extract_actions_from_tool_calls
from src.tiers import get_tiers

logger = logging.getLogger(__name__)


async def handle_generate(msg: Msg) -> None:
    """
    Subject: mordomo.brain.generate
    Expects: { text, speaker_id, person_id?, confidence? }
    Replies: { response_text, model_used, tokens_used, actions, tier, intents }

    Flow:
      1. Load conversation history + RAG context
      2. Phase 1 — classify risk tier + intents via Bifrost/Groq
      3. Phase 2 — Stream response from LLM, publishing each sentence to TTS
      4. Publish actions to NATS, reply to orchestrator with full response
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

    # 1. Load conversation history and RAG context
    history     = await get_history(speaker_id)
    rag_context = await search(text, speaker_id)

    # 2. Phase 1 — classify tier
    tiers = await get_tiers()
    classification = await classify(text, list(tiers.keys()))

    # 3. Phase 2 — stream response, publishing sentences to TTS as they arrive
    full_text_parts = []
    model_used = ""
    tokens_used = 0
    tool_calls = []
    sentence_idx = 0

    try:
        async for chunk in generate_stream(text, history, rag_context, tier=classification.tier):
            if chunk.text:
                full_text_parts.append(chunk.text)

                # Publish sentence to TTS for immediate synthesis
                tts_payload = {
                    "text": chunk.text,
                    "sentence_index": sentence_idx,
                    "is_final": chunk.is_final,
                    "timestamp": time.time(),
                }
                await nc.publish(
                    f"mordomo.tts.stream.{speaker_id}",
                    json.dumps(tts_payload).encode(),
                )
                sentence_idx += 1

            if chunk.is_final:
                model_used = chunk.model_used
                tokens_used = chunk.tokens_used
                tool_calls = chunk.tool_calls or []

    except Exception as exc:
        logger.error("LLM streaming failed: %s", exc)
        if msg.reply:
            await nc.publish(msg.reply, json.dumps({"error": "llm_failed", "detail": str(exc)}).encode())
        return

    clean_text = " ".join(full_text_parts)

    # 4. Extract actions from tool_calls
    actions = extract_actions_from_tool_calls(tool_calls)

    # 5. Persist exchange to Redis context
    await append_exchange(speaker_id, text, clean_text)

    # 6. Store exchange in Qdrant for future RAG
    await upsert(
        text=f"Usuário: {text}\nMordomo: {clean_text}",
        speaker_id=speaker_id,
        point_id=str(uuid.uuid4()),
    )

    # 7. Publish each detected action to NATS
    for action in actions:
        action_type = action.get("type", "generic")
        subject = f"{SUBJECT_ACTION}.{action_type}"
        payload = {**action, "speaker_id": speaker_id}
        await nc.publish(subject, json.dumps(payload).encode())
        logger.info("Action published: %s → %s", subject, payload)

    # 8. Reply to orchestrator with full response
    if msg.reply:
        reply_payload = {
            "response_text": clean_text,
            "model_used":    model_used,
            "tokens_used":   tokens_used,
            "actions":       actions,
            "tier":          classification.tier,
            "intents":       classification.intents,
        }
        await nc.publish(msg.reply, json.dumps(reply_payload).encode())

    logger.info(
        "Generated [speaker=%s tier=%s model=%s tokens=%d actions=%d intents=%s sentences=%d]",
        speaker_id, classification.tier, model_used,
        tokens_used, len(actions), classification.intents, sentence_idx,
    )
