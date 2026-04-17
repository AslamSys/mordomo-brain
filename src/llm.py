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
import re
from dataclasses import dataclass, field
from typing import AsyncIterator

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
    Phase 1 — Classify risk tier and intents via Bifrost (Groq backend).
    Falls back to direct Groq if Bifrost is unavailable, then to TIER_FALLBACK.
    """
    user_prompt = (
        f"Available tiers: {tiers}\n\n"
        f"User message: \"{text}\"\n\n"
        "Classify and respond with JSON only."
    )
    messages = [
        {"role": "system", "content": _CLASSIFY_SYSTEM},
        {"role": "user",   "content": user_prompt},
    ]
    payload = {
        "model": GROQ_MODEL,
        "messages": messages,
        "max_tokens": 128,
        "temperature": 0,
        "response_format": {"type": "json_object"},
    }

    # 1. Try Bifrost gateway (routes to Groq via config)
    try:
        headers = {"Content-Type": "application/json"}
        if BIFROST_API_KEY:
            headers["Authorization"] = f"Bearer {BIFROST_API_KEY}"

        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(
                f"{BIFROST_URL}/v1/chat/completions",
                headers=headers,
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()
            raw = data["choices"][0]["message"]["content"]
            parsed = json.loads(raw)

            tier = parsed.get("tier", TIER_FALLBACK)
            intents = parsed.get("intents", [])

            if tier not in tiers:
                logger.warning("Classifier returned unknown tier '%s', falling back to '%s'", tier, TIER_FALLBACK)
                tier = TIER_FALLBACK

            logger.info("Classified via Bifrost [tier=%s intents=%s]", tier, intents)
            return ClassifyResult(tier=tier, intents=intents)

    except Exception as exc:
        logger.warning("Bifrost classify failed: %s — trying Groq direct", exc)

    # 2. Fallback: direct Groq
    if not GROQ_API_KEY:
        logger.warning("GROQ_API_KEY not set — using fallback tier")
        return ClassifyResult(tier=TIER_FALLBACK, intents=["unknown"])

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(
                f"{GROQ_URL}/chat/completions",
                headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()
            raw = data["choices"][0]["message"]["content"]
            parsed = json.loads(raw)

            tier = parsed.get("tier", TIER_FALLBACK)
            intents = parsed.get("intents", [])

            if tier not in tiers:
                logger.warning("Classifier returned unknown tier '%s', falling back to '%s'", tier, TIER_FALLBACK)
                tier = TIER_FALLBACK

            logger.info("Classified via Groq direct [tier=%s intents=%s]", tier, intents)
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


# ── Sentence splitter for streaming ────────────────────────────────────────

_SENTENCE_END = re.compile(r'(?<=[.!?…])\s+|(?<=\n)')


@dataclass
class StreamChunk:
    """A sentence or partial text emitted during streaming."""
    text: str
    is_final: bool = False
    tool_calls: list[dict] | None = None
    model_used: str = ""
    tokens_used: int = 0


async def generate_stream(
    user_text: str,
    history: list[dict],
    rag_context: list[str],
    tier: str,
) -> AsyncIterator[StreamChunk]:
    """
    Phase 2 (streaming) — Stream response sentence by sentence via SSE.
    Yields StreamChunk for each complete sentence. Final chunk has is_final=True
    and includes tool_calls + usage metadata.

    Falls back to non-streaming generate() if SSE fails.
    """
    messages = _build_messages(user_text, history, rag_context)

    model, fallback_model = await resolve_tier(tier)
    if not model:
        raise RuntimeError(f"No model configured for tier '{tier}'")

    tools = await get_tools()
    headers = {"Content-Type": "application/json"}
    if BIFROST_API_KEY:
        headers["Authorization"] = f"Bearer {BIFROST_API_KEY}"

    payload = {
        "model": model,
        "messages": messages,
        "tools": tools,
        "tool_choice": "auto",
        "max_tokens": 1024,
        "temperature": 0.7,
        "stream": True,
    }
    if fallback_model and fallback_model != model:
        payload["fallbacks"] = [fallback_model]

    try:
        async with httpx.AsyncClient(timeout=60) as client:
            async with client.stream(
                "POST",
                f"{BIFROST_URL}/v1/chat/completions",
                headers=headers,
                json=payload,
            ) as resp:
                resp.raise_for_status()
                buffer = ""
                all_text = ""
                tool_call_deltas: dict[int, dict] = {}
                model_used = model
                tokens_used = 0

                async for line in resp.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    data_str = line[6:]
                    if data_str.strip() == "[DONE]":
                        break

                    try:
                        data = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue

                    # Track usage if present
                    if "usage" in data:
                        tokens_used = data["usage"].get("total_tokens", tokens_used)
                    if "model" in data:
                        model_used = data["model"]

                    delta = data.get("choices", [{}])[0].get("delta", {})

                    # Accumulate tool call deltas
                    for tc in delta.get("tool_calls", []):
                        idx = tc.get("index", 0)
                        if idx not in tool_call_deltas:
                            tool_call_deltas[idx] = {"function": {"name": "", "arguments": ""}}
                        fn = tc.get("function", {})
                        if "name" in fn:
                            tool_call_deltas[idx]["function"]["name"] = fn["name"]
                        if "arguments" in fn:
                            tool_call_deltas[idx]["function"]["arguments"] += fn["arguments"]

                    # Accumulate text content
                    content = delta.get("content", "")
                    if not content:
                        continue

                    buffer += content
                    all_text += content

                    # Split on sentence boundaries and yield complete sentences
                    parts = _SENTENCE_END.split(buffer)
                    if len(parts) > 1:
                        # All except last are complete sentences
                        for sentence in parts[:-1]:
                            sentence = sentence.strip()
                            if sentence:
                                yield StreamChunk(text=sentence, model_used=model_used)
                        buffer = parts[-1]

                # Yield remaining buffer as final chunk
                final_text = buffer.strip()
                tool_calls = []
                for _idx, tc in sorted(tool_call_deltas.items()):
                    fn = tc["function"]
                    try:
                        args = json.loads(fn["arguments"])
                    except json.JSONDecodeError:
                        args = {}
                    tool_calls.append({"type": fn["name"], **args})

                yield StreamChunk(
                    text=final_text,
                    is_final=True,
                    tool_calls=tool_calls,
                    model_used=model_used,
                    tokens_used=tokens_used,
                )
                return

    except Exception as exc:
        logger.warning("Streaming failed: %s — falling back to non-streaming", exc)

    # Fallback: non-streaming generate
    resp = await generate(user_text, history, rag_context, tier)
    yield StreamChunk(
        text=resp.text,
        is_final=True,
        tool_calls=[{"type": tc.get("type", "unknown"), **{k: v for k, v in tc.items() if k != "type"}} for tc in resp.tool_calls],
        model_used=resp.model_used,
        tokens_used=resp.tokens_used,
    )
