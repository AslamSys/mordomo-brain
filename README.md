# mordomo-brain

## Proposito

Core cognitivo do Mordomo. Recebe texto via NATS, classifica risco/complexidade (fase 1), gera resposta com tools (fase 2), persiste contexto e publica acoes estruturadas.

## Arquitetura atual

### Fase 1 - Classificacao (Bifrost → Groq)

- Endpoint primario: `BIFROST_URL/v1/chat/completions` (roteia para Groq)
- Fallback: Groq direto (`https://api.groq.com/openai/v1/chat/completions`)
- Modelo padrao: `llama-3.3-70b-versatile`
- Saida: `tier` semantico (`simple`, `brain`, `stakes`) + `intents`

### Fase 2 - Geracao (Bifrost, com streaming)

- Endpoint: `BIFROST_URL/v1/chat/completions` (SSE streaming)
- O brain resolve `tier` -> `provider/model` consultando Redis db1
- Streaming: frases sao publicadas via NATS (`mordomo.tts.stream.{speaker_id}`) conforme chegam do LLM
- Fallback por request enviado no payload (`fallbacks`)
- Em falha total do gateway, usa Groq direto em modo degradado (sem function calling, sem streaming)

### RAG (Qdrant + embeddings via Bifrost)

- Embedding endpoint: `BIFROST_URL/v1/embeddings`
- Modelo de embedding: `EMBEDDING_MODEL` (default `gemini/text-embedding-004`)
- Colecao Qdrant: `QDRANT_COLLECTION` (default `mordomo_conversations`)

## Redis db1

### Chaves usadas pelo brain

- `brain:context:{speaker_id}`: historico da conversa
- `mordomo:tools`: registry de tools (lido pelo brain)
- `mordomo:tiers`: mapping tier semantico -> provider/model
- `mordomo:tiers:fallbacks`: mapping tier semantico -> provider/model fallback

Observacao importante:

- O seed de `mordomo:tiers*` nao e responsabilidade do brain.
- O seed e feito pelo fluxo de deploy (`mordomo-deploy/infra/redis/seed-brain-tiers.sh`).

## NATS

### Entrada

- Subject: `mordomo.brain.generate`
- Payload esperado:

```json
{
  "speaker_id": "user_1",
  "text": "acende a luz da sala",
  "confidence": 0.97
}
```

### Saida

- Reply: `response_text`, `model_used`, `tokens_used`, `actions`, `tier`, `intents`
- Publish de acoes: `mordomo.brain.action.{type}`
- Streaming de frases: `mordomo.tts.stream.{speaker_id}` (cada frase conforme chega do LLM)

## Variaveis de ambiente relevantes

- `NATS_URL`
- `REDIS_URL`
- `QDRANT_URL`
- `QDRANT_COLLECTION`
- `QDRANT_VECTOR_SIZE` (default 768)
- `RAG_ENABLED`
- `RAG_TOP_K`
- `RAG_MIN_SCORE`
- `BIFROST_URL`
- `BIFROST_API_KEY`
- `EMBEDDING_MODEL`
- `GROQ_API_KEY`
- `GROQ_MODEL`
- `TIER_FALLBACK`
- `TIER_CACHE_TTL`
- `TIER_STRICT_MODE`
- `TOOLS_CACHE_TTL`

## Fluxo resumido

1. `handle_generate` recebe texto e speaker
2. Busca historico (Redis) e contexto RAG (Qdrant)
3. Classifica tier/intents com Groq
4. Resolve tier no Redis para modelo real
5. Chama Bifrost com tools + fallback model
6. Persiste contexto e upsert no Qdrant
7. Publica acoes em `mordomo.brain.action.*`
8. Responde ao solicitante via reply NATS
