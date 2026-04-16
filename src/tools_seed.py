"""
Built-in tool definitions — seeded into Redis on brain startup.

These are the defaults that work out-of-the-box with the core services.
A service can OVERRIDE any of these by writing its own definition:
    HSET mordomo:tools <name> '<json>'

A service can EXTEND the registry with new capabilities:
    HSET mordomo:tools my_new_capability '<json>'

The brain discovers them automatically within TOOLS_CACHE_TTL seconds.

Keys must match the function `name` field inside the definition.
"""

SEED_TOOLS: dict[str, dict] = {
    "iot_control": {
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
    "pix_send": {
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
    "balance_query": {
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
    "reminder_create": {
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
    "alarm_control": {
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
    "openclaw_execute": {
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
    "media_control": {
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
}
