"""
NATS handler for mordomo.brain.generate (request/reply).
"""
import json
import logging
import uuid

from nats.aio.msg import Msg

from src.config import SUBJECT_ACTION
from src.context import get_history, append_exchange
from src.llm import generate
from src.rag import search, upsert
from src.actions import extract_actions

logger = logging.getLogger(__name__)


async def handle_generate(msg: Msg) -> None:
    """
    Subject: mordomo.brain.generate
    Expects: { text, speaker_id, person_id?, confidence? }
    Replies: { response_text, model_used, tokens_used, actions }
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

    # 1. Load conversation history from Redis
    history = await get_history(speaker_id)

    # 2. RAG — search semantically relevant past context
    rag_context = await search(text, speaker_id)

    # 3. Call LLM via LiteLLM gateway
    try:
        llm_resp = await generate(text, history, rag_context, speaker_id)
    except Exception as exc:
        logger.error("LLM generation failed: %s", exc)
        if msg.reply:
            await nc.publish(msg.reply, json.dumps({"error": "llm_failed", "detail": str(exc)}).encode())
        return

    # 4. Extract action annotations from response
    clean_text, actions = extract_actions(llm_resp.text)

    # 5. Persist exchange to Redis context
    await append_exchange(speaker_id, text, clean_text)

    # 6. Store exchange in Qdrant for future RAG
    await upsert(
        text=f"Usuário: {text}\nMordomo: {clean_text}",
        speaker_id=speaker_id,
        point_id=str(uuid.uuid4()),
    )

    # 7. Publish detected actions
    for action in actions:
        action_type = action.get("type", "generic")
        subject = f"{SUBJECT_ACTION}.{action_type}"
        payload = {**action, "speaker_id": speaker_id}
        await nc.publish(subject, json.dumps(payload).encode())
        logger.info("Action published: %s → %s", subject, payload)

    # 8. Reply to caller (orchestrator)
    if msg.reply:
        reply_payload = {
            "response_text": clean_text,
            "model_used":    llm_resp.model_used,
            "tokens_used":   llm_resp.tokens_used,
            "actions":       actions,
        }
        await nc.publish(msg.reply, json.dumps(reply_payload).encode())

    logger.info(
        "Generated [speaker=%s model=%s tokens=%d actions=%d]",
        speaker_id, llm_resp.model_used, llm_resp.tokens_used, len(actions)
    )
