import os

NATS_URL: str = os.getenv("NATS_URL", "nats://nats:4222")

# LiteLLM gateway (já configurado na infra com as API keys)
LITELLM_URL: str = os.getenv("LITELLM_URL", "http://mordomo-litellm:4000")

# Groq — fallback direto, bypassa o gateway quando necessário
# Só é chamado se o LiteLLM gateway falhar completamente ou não tiver chaves.
GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
GROQ_URL: str     = "https://api.groq.com/openai/v1"
GROQ_MODEL: str   = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

# Redis db1 (mordomo-general)
REDIS_URL: str = os.getenv("REDIS_URL", "redis://mordomo-redis:6379/1")

# Qdrant (RAG)
QDRANT_URL: str        = os.getenv("QDRANT_URL",        "http://mordomo-qdrant:6333")
QDRANT_COLLECTION: str = os.getenv("QDRANT_COLLECTION", "mordomo_conversations")
QDRANT_VECTOR_SIZE: int = int(os.getenv("QDRANT_VECTOR_SIZE", "1536"))  # text-embedding-3-small
RAG_ENABLED: bool  = os.getenv("RAG_ENABLED", "true").lower() == "true"
RAG_TOP_K: int     = int(os.getenv("RAG_TOP_K",      "4"))
RAG_MIN_SCORE: float = float(os.getenv("RAG_MIN_SCORE", "0.70"))

# NATS subjects
SUBJECT_GENERATE = "mordomo.brain.generate"
SUBJECT_ACTION   = "mordomo.brain.action"  # + ".{type}"

# LLM virtual model names (configurados no LiteLLM gateway)
MODEL_SIMPLE:  str = os.getenv("MODEL_SIMPLE",  "mordomo-simple")   # gemini-2.0-flash
MODEL_COMPLEX: str = os.getenv("MODEL_COMPLEX", "mordomo-complex")  # claude-haiku
MODEL_STAKES:  str = os.getenv("MODEL_STAKES",  "mordomo-stakes")   # claude-sonnet

# Context window
CONTEXT_MAX_MESSAGES: int    = int(os.getenv("CONTEXT_MAX_MESSAGES",    "12"))
CONTEXT_TTL_SECONDS:  int    = int(os.getenv("CONTEXT_TTL_SECONDS",     "1800"))  # 30 min
CONTEXT_SUMMARIZE_KEEP: int  = int(os.getenv("CONTEXT_SUMMARIZE_KEEP",  "4"))     # msgs recentes preservadas verbatim

# System prompt
SYSTEM_PROMPT: str = os.getenv("SYSTEM_PROMPT", """Você é Mordomo, um assistente doméstico inteligente.
Você controla dispositivos IoT, responde perguntas e ajuda os moradores.
Seja conciso, útil e natural nas respostas. Responda sempre em português brasileiro.
Quando identificar uma ação (IoT, lembrete, finanças), inclua no final da resposta um JSON de ação:
[ACTION: {"type": "iot_control", "device": "...", "command": "..."}]
""")
