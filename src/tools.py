"""
Dynamic tool registry for mordomo-brain.

Tools are NOT hardcoded here. They are stored in Redis (HSET mordomo:tools <name> <json>)
and discovered at startup + refreshed every TOOLS_CACHE_TTL seconds.

How to add a new capability:
  1. In your service's startup, publish the tool definition to Redis:
       HSET mordomo:tools my_new_tool '{"type":"function","function":{...}}'
  2. Add the NATS subject mapping in mordomo-orchestrator/src/dispatcher.py
  That's all — no changes needed in the brain.

The brain uses get_tools() in llm.py (phase 2 generation).
On startup, init_tools() seeds Redis with built-in defaults if the key doesn't exist yet,
so the system works out of the box without requiring all services to be up first.
"""
import json
import logging
import time

import redis.asyncio as aioredis

from src.config import REDIS_URL, TOOLS_CACHE_TTL
from src.tools_seed import SEED_TOOLS

logger = logging.getLogger(__name__)

REDIS_KEY = "mordomo:tools"

# In-memory cache (same pattern as tiers.py)
_cached_tools: list[dict] = []
_cache_ts: float = 0.0


async def _redis() -> aioredis.Redis:
    return await aioredis.from_url(REDIS_URL, decode_responses=True)


async def fetch_tools() -> list[dict]:
    """
    Fetch all registered tools from Redis HSET mordomo:tools.
    Returns the seed defaults on failure.
    """
    try:
        r = await _redis()
        raw = await r.hgetall(REDIS_KEY)
        await r.aclose()

        if not raw:
            logger.warning("No tools found in Redis '%s' — using seed defaults", REDIS_KEY)
            return list(SEED_TOOLS.values())

        tools = []
        for name, value in raw.items():
            try:
                tools.append(json.loads(value))
            except json.JSONDecodeError:
                logger.warning("Skipping malformed tool definition for '%s'", name)

        logger.info("Loaded %d tools from Redis", len(tools))
        return tools

    except Exception as exc:
        logger.warning("Failed to fetch tools from Redis: %s — using seed defaults", exc)
        return list(SEED_TOOLS.values())


async def get_tools() -> list[dict]:
    """
    Returns cached tool list, refreshing if TTL has expired.
    Call this in llm.py phase 2 generate().
    """
    global _cached_tools, _cache_ts
    now = time.monotonic()

    if not _cached_tools or (now - _cache_ts) > TOOLS_CACHE_TTL:
        _cached_tools = await fetch_tools()
        _cache_ts = now

    return _cached_tools


async def init_tools() -> None:
    """
    Called once at brain startup.
    Seeds Redis with built-in defaults ONLY for tools not already registered.
    Services that register their own definitions override these naturally.
    """
    try:
        r = await _redis()
        for name, definition in SEED_TOOLS.items():
            # HSETNX — only sets if the field doesn't already exist
            await r.hsetnx(REDIS_KEY, name, json.dumps(definition))
        await r.aclose()
        logger.info("Tool registry seeded (%d built-in tools)", len(SEED_TOOLS))
    except Exception as exc:
        logger.warning("Could not seed tool registry: %s — will use in-memory defaults", exc)

    # Warm the in-memory cache immediately
    await get_tools()


# ── Legacy compatibility ───────────────────────────────────────────────────
# llm.py imports TOOLS directly in some paths. This shim keeps it working
# until all callers are updated to use get_tools() (async).
# It returns the current cache — will be empty list before init_tools() runs.
@property
def TOOLS() -> list[dict]:  # type: ignore[override]
    return _cached_tools


# Built-in defaults kept as SEED_TOOLS in tools_seed.py — not here.
_PLACEHOLDER: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "iot_control",
            "description": (
                "Controls physical devices in the home: lights, outlets, TV, air conditioning, "
                "gate, blinds, fans, smart plugs. Use when the user wants to turn something on/off "
                "or adjust a device state."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "device_id": {
                        "type": "string",
                        "description": "Device identifier, e.g. luz_sala, ar_quarto, portao, tv_sala",
                    },
                    "command": {
                        "type": "string",
                        "enum": ["on", "off", "toggle", "set"],
                        "description": "Command to execute",
                    },
                    "value": {
                        "type": "string",
                        "description": "Optional value for 'set' command, e.g. '22' for temperature, '50%' for brightness",
                    },
                },
                "required": ["device_id", "command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "pix_send",
            "description": (
                "Sends money via PIX to a person or key. Use ONLY when the user explicitly "
                "wants to transfer or pay money — NOT for balance queries."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "recipient": {
                        "type": "string",
                        "description": "Person name or PIX key (CPF, phone, email, random key)",
                    },
                    "amount": {
                        "type": "number",
                        "description": "Amount in BRL (Brazilian Reais)",
                    },
                    "description": {
                        "type": "string",
                        "description": "Optional transfer description",
                    },
                },
                "required": ["recipient", "amount"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "balance_query",
            "description": (
                "Queries bank account balance, recent transactions, or financial summary. "
                "Use when the user asks 'how much do I have', 'what's my balance', "
                "'show my transactions', etc. Does NOT move money."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "account_type": {
                        "type": "string",
                        "description": "Account type: checking, savings, pix, investments. Omit for all accounts.",
                    },
                    "period": {
                        "type": "string",
                        "description": "Optional period, e.g. 'last 7 days', 'this month'",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "reminder_create",
            "description": "Creates a reminder, alarm or scheduled notification for the user.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "What to remind about",
                    },
                    "datetime": {
                        "type": "string",
                        "description": "When to remind — ISO 8601 or natural language like 'tomorrow 8am', 'in 30 minutes'",
                    },
                    "repeat": {
                        "type": "string",
                        "description": "Optional recurrence: daily, weekly, weekdays",
                    },
                },
                "required": ["text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "alarm_control",
            "description": (
                "Arms, disarms or queries the home security alarm system. "
                "High-sensitivity action — requires vault authorization."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["arm", "disarm", "status", "panic"],
                        "description": "Action to perform on the alarm",
                    },
                    "zone": {
                        "type": "string",
                        "description": "Optional zone: perimeter, internal, garage. Omit for full system.",
                    },
                },
                "required": ["action"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "openclaw_execute",
            "description": (
                "Executes browser automation or messaging tasks via the OpenClaw agent. "
                "Use for: sending WhatsApp/Telegram messages, web browsing, filling forms, "
                "opening apps, or any task requiring computer/browser interaction."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "task": {
                        "type": "string",
                        "description": "Natural language description of the task to execute",
                    },
                    "target": {
                        "type": "string",
                        "description": "Target app or platform: whatsapp, telegram, browser, system",
                    },
                    "payload": {
                        "type": "object",
                        "description": "Optional structured data for the task, e.g. {recipient, message}",
                    },
                },
                "required": ["task"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "media_control",
            "description": (
                "Controls media playback on TV, speakers or streaming services. "
                "Use for play, pause, volume, channel changes, and content requests."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "device": {
                        "type": "string",
                        "description": "Target device: tv, speakers, tv_sala, tv_quarto",
                    },
                    "action": {
                        "type": "string",
                        "enum": ["play", "pause", "stop", "next", "previous", "volume_up", "volume_down", "mute", "unmute", "open"],
                        "description": "Media action",
                    },
                    "content": {
                        "type": "string",
                        "description": "Optional content to play or open, e.g. 'Netflix', 'YouTube', song name",
                    },
                },
                "required": ["device", "action"],
            },
        },
    },
]
# ── end of _PLACEHOLDER (this list is IGNORED at runtime) ─────────────────
# The actual tools list is always fetched from Redis via get_tools().
# _PLACEHOLDER is kept only as documentation of the built-in defaults.
# The canonical definitions live in tools_seed.py.
