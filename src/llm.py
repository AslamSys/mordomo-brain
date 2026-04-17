"""
LLM client — 2-phase architecture.

Phase 1 — Classification (Groq, always, free):
    Receives the user text + available semantic tiers.
    Returns: { tier: "stakes", intents: ["pix_send", "iot_control"] }
  This is semantic classification — no keyword matching.

Phase 2 — Generation (chosen tier via Bifrost gateway, with function calling):
  Receives text + history + RAG context + tools definition.
  Returns: response_text + tool_calls (structured actions, no regex parsing needed).

Fallback chain for phase 2:
    1. Bifrost gateway (resolved model + per-request fallback)
  2. Groq direct without tools (last resort — graceful degradation)
"""
import json
import logging
from dataclasses import dataclass, field

import httpx

from src.config import (
    BIFROST_URL, BIFROST_API_KEY,
    GROQ_API_KEY, GROQ_URL, GROQ_MODEL,
    SYSTEM_PROMPT, TIER_FALLBACK,
)
from src.tiers import resolve_tier
from src.tools import get_tools

logger = logging.getLogger(__name__)


@dataclass
class ClassifyResult:
    tier: str
    intents: list[str]


@dataclass
class LLMResponse:
    text: str
    model_used: str
    tokens_used: int
    tool_calls: list[dict] = field(default_factory=list)


# ── Phase 1: Classification ────────────────────────────────────────────────

_CLASSIFY_SYSTEM = """You are a classifier for a smart home assistant called Mordomo.
Given a user message and a list of available LLM tiers, determine:
1. Which tier should handle this request (choose the minimum necessary tier)
2. Which intents/actions are present

Tier guidance:
- simple: short commands, IoT control, reminders, quick questions
- brain: multi-step reasoning, analysis, long context, comparisons
- stakes: financial transactions (sending money), security credentials, alarms, anything irreversible

Rules:
- When multiple intents are present, use the tier of the HIGHEST risk intent
- Balance queries are NOT stakes — only actual money transfers are
- Always respond with valid JSON only, no explanation

Response format: {"tier": "<tier_name>", "intents": ["intent1", "intent2"]}"""


async def classify(text: str, tiers: list[str]) -> ClassifyResult:
    """
    Phase 1 — Call Groq directly to classify risk tier and intents.
    Groq is always used here: free, fast (~200ms), no gateway dependency.
    Falls back to TIER_FALLBACK if Groq is unavailable.
    """
    if not GROQ_API_KEY:
        logger.warning("GROQ_API_KEY not set — skipping classification, using fallback tier")
        return ClassifyResult(tier=TIER_FALLBACK, intents=["unknown"])

    user_prompt = (
        f"Available tiers: {tiers}\n\n"
        f"User message: \"{text}\"\n\n"
        "Classify and respond with JSON only."
    )

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(
                f"{GROQ_URL}/chat/completions",
                headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
                json={
                    "model": GROQ_MODEL,
                    "messages": [
                        {"role": "system", "content": _CLASSIFY_SYSTEM},
                        {"role": "user",   "content": user_prompt},
                    ],
                    "max_tokens": 128,
                    "temperature": 0,
                    "response_format": {"type": "json_object"},
                },
            )
            resp.raise_for_status()
            data = resp.json()
            raw = data["choices"][0]["message"]["content"]
            parsed = json.loads(raw)

            tier = parsed.get("tier", TIER_FALLBACK)
            intents = parsed.get("intents", [])

            # Safety: tier must be one of the known tiers
            if tier not in tiers:
                logger.warning("Classifier returned unknown tier '%s', falling back to '%s'", tier, TIER_FALLBACK)
                tier = TIER_FALLBACK

            logger.info("Classified [tier=%s intents=%s]", tier, intents)
            return ClassifyResult(tier=tier, intents=intents)

    except Exception as exc:
        logger.warning("Classification failed: %s — using fallback tier '%s'", exc, TIER_FALLBACK)
        return ClassifyResult(tier=TIER_FALLBACK, intents=["unknown"])


# ── Phase 2: Generation ────────────────────────────────────────────────────

def _build_messages(user_text: str, history: list[dict], rag_context: list[str]) -> list[dict]:
    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]

    if rag_context:
        context_block = "\n".join(f"- {c}" for c in rag_context)
        messages.append({
            "role": "system",
            "content": f"Contexto relevante de conversas anteriores:\n{context_block}",
        })

    messages.extend(history)
    messages.append({"role": "user", "content": user_text})
    return messages


def _extract_tool_calls(choice: dict) -> list[dict]:
    """Extract tool_calls from a chat completion choice into a flat list of action dicts."""
    raw_calls = choice.get("message", {}).get("tool_calls") or []
    actions = []
    for tc in raw_calls:
        fn = tc.get("function", {})
        try:
            args = json.loads(fn.get("arguments", "{}"))
        except json.JSONDecodeError:
            args = {}
        actions.append({"type": fn.get("name", "unknown"), **args})
    return actions


async def _call_bifrost_with_tools(model: str, fallback_model: str, messages: list[dict]) -> LLMResponse:
    """Call Bifrost gateway with function calling enabled. Raises on failure."""
    headers = {"Content-Type": "application/json"}
    if BIFROST_API_KEY:
        headers["Authorization"] = f"Bearer {BIFROST_API_KEY}"

    tools = await get_tools()
    payload = {
        "model": model,
        "messages": messages,
        "tools": tools,
        "tool_choice": "auto",
        "max_tokens": 1024,
        "temperature": 0.7,
    }
    if fallback_model and fallback_model != model:
        payload["fallbacks"] = [fallback_model]

    async with httpx.AsyncClient(timeout=45) as client:
        resp = await client.post(
            f"{BIFROST_URL}/v1/chat/completions",
            headers=headers,
            json=payload,
        )
        resp.raise_for_status()
        data = resp.json()

    choice = data["choices"][0]
    content = choice.get("message", {}).get("content") or ""
    tokens = data.get("usage", {}).get("total_tokens", 0)
    tool_calls = _extract_tool_calls(choice)

    return LLMResponse(text=content, model_used=model, tokens_used=tokens, tool_calls=tool_calls)


async def _call_groq_fallback(messages: list[dict]) -> LLMResponse:
    """
    Last-resort fallback — calls Groq directly without function calling.
    Produces a text-only response with no structured actions.
    Used only when the Bifrost gateway is completely unreachable.
    """
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY not set — no fallback available")

    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            f"{GROQ_URL}/chat/completions",
            headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
            json={
                "model": GROQ_MODEL,
                "messages": messages,
                "max_tokens": 512,
                "temperature": 0.7,
            },
        )
        resp.raise_for_status()
        data = resp.json()

    content = data["choices"][0]["message"]["content"] or ""
    tokens = data.get("usage", {}).get("total_tokens", 0)
    logger.warning("Used Groq fallback — no function calling, actions will be empty")
    return LLMResponse(text=content, model_used=f"groq/{GROQ_MODEL}", tokens_used=tokens)


async def generate(
    user_text: str,
    history: list[dict],
    rag_context: list[str],
    tier: str,
) -> LLMResponse:
    """
    Phase 2 — Generate response using the tier chosen by classify().
    Function calling is used to extract structured actions from the LLM response.
    """
    messages = _build_messages(user_text, history, rag_context)

    model, fallback_model = await resolve_tier(tier)
    if not model:
        raise RuntimeError(f"No model configured for tier '{tier}' and no valid fallback tier '{TIER_FALLBACK}'")

    # 1. Bifrost gateway (with function calling + per-request fallbacks)
    try:
        return await _call_bifrost_with_tools(model, fallback_model, messages)
    except Exception as exc:
        logger.warning("Bifrost gateway failed [tier=%s model=%s]: %s — trying Groq direct", tier, model, exc)

    # 2. Groq direct — degraded mode, no structured actions
    return await _call_groq_fallback(messages)
