import os

NATS_URL: str = os.getenv("NATS_URL", "nats://nats:4222")

# LiteLLM gateway
LITELLM_URL: str        = os.getenv("LITELLM_URL",        "http://llm-gateway:4000")
LITELLM_MASTER_KEY: str = os.getenv("LITELLM_MASTER_KEY", "")

# Groq — usado como classificador de tier (fase 1, sempre) e fallback de emergência
GROQ_API_KEY: str   = os.getenv("GROQ_API_KEY", "")
GROQ_URL: str       = "https://api.groq.com/openai/v1"
GROQ_MODEL: str     = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

# Redis db1 (mordomo-general)
REDIS_URL: str = os.getenv("REDIS_URL", "redis://redis:6379/1")

# Qdrant (RAG)
QDRANT_URL: str         = os.getenv("QDRANT_URL",         "http://qdrant:6333")
QDRANT_COLLECTION: str  = os.getenv("QDRANT_COLLECTION",  "mordomo_conversations")
QDRANT_VECTOR_SIZE: int = int(os.getenv("QDRANT_VECTOR_SIZE", "1536"))  # text-embedding-3-small
RAG_ENABLED: bool       = os.getenv("RAG_ENABLED", "true").lower() == "true"
RAG_TOP_K: int          = int(os.getenv("RAG_TOP_K",       "4"))
RAG_MIN_SCORE: float    = float(os.getenv("RAG_MIN_SCORE", "0.70"))

# NATS subjects
SUBJECT_GENERATE = "mordomo.brain.generate"
SUBJECT_ACTION   = "mordomo.brain.action"  # + ".{type}"

# Tool registry — tools dinâmicas via Redis HSET mordomo:tools
TOOLS_CACHE_TTL: int = int(os.getenv("TOOLS_CACHE_TTL", "120"))  # segundos — re-descobre a cada 2 min

# Tier discovery — o brain consulta o gateway dinamicamente
# TIER_PREFIX filtra quais model_names do gateway são "tiers do brain"
TIER_PREFIX: str    = os.getenv("TIER_PREFIX",    "mordomo-")
TIER_CACHE_TTL: int = int(os.getenv("TIER_CACHE_TTL", "300"))  # segundos — re-descobre a cada 5 min
# Tier de fallback usado quando a descoberta falha completamente no startup
TIER_FALLBACK: str  = os.getenv("TIER_FALLBACK",  "mordomo-simple")

# Context window
CONTEXT_MAX_MESSAGES: int   = int(os.getenv("CONTEXT_MAX_MESSAGES",   "12"))
CONTEXT_TTL_SECONDS: int    = int(os.getenv("CONTEXT_TTL_SECONDS",    "1800"))  # 30 min
CONTEXT_SUMMARIZE_KEEP: int = int(os.getenv("CONTEXT_SUMMARIZE_KEEP", "4"))

# System prompt — não instrui o LLM a escrever tags; ações vêm via function calling
SYSTEM_PROMPT: str = os.getenv("SYSTEM_PROMPT", """Você é Mordomo, assistente doméstico inteligente do Renan.
Você controla dispositivos, gerencia finanças, cria lembretes e ajuda com qualquer tarefa doméstica.
Seja conciso, natural e proativo. Responda sempre em português brasileiro.
Quando identificar uma ação a executar, use as ferramentas disponíveis (function calling).
Você pode chamar múltiplas ferramentas em uma única resposta quando necessário.
""")
