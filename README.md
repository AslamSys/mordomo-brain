# 🧠 mordomo-brain

## 🔗 Navegação

**[🏠 AslamSys](https://github.com/AslamSys)** → **[📚 _system](https://github.com/AslamSys/_system)** → **[📂 Aslam (Orange Pi 5 16GB)](https://github.com/AslamSys/_system/blob/main/hardware/mordomo%20-%20(orange-pi-5-16gb)/README.md)** → **mordomo-brain**

### Containers Relacionados (aslam)
- [mordomo-orchestrator](https://github.com/AslamSys/mordomo-orchestrator)
- [mordomo-tts-engine](https://github.com/AslamSys/mordomo-tts-engine)
- [mordomo-people](https://github.com/AslamSys/mordomo-people)
- [mordomo-vault](https://github.com/AslamSys/mordomo-vault)
- [infra/llm-gateway](https://github.com/AslamSys/mordomo-deploy) — LiteLLM proxy
- [infra/redis](https://github.com/AslamSys/mordomo-deploy) — db1 (sessions + tools + routes)
- [infra/qdrant](https://github.com/AslamSys/mordomo-deploy) — RAG

---

**Container:** `mordomo-brain`  
**Ecossistema:** Brain  
**Hardware:** Orange Pi 5 Ultra  
**Linguagem:** Python 3.11 (asyncio + nats-py)

---

## 📋 Propósito

LLM cognitive core do Mordomo. Recebe texto transcrito via NATS (request/reply), executa classificação em 2 fases, retorna resposta + ações estruturadas.

---

## 🎯 Responsabilidades

- Classificar intenção e tier de complexidade (Fase 1 — Groq, ~200ms)
- Gerar resposta com function calling (Fase 2 — LiteLLM gateway)
- Carregar ferramentas disponíveis do Redis (`mordomo:tools`)
- Descobrir tiers disponíveis do LiteLLM (`/models`)
- Manter contexto de conversa por speaker (Redis db1, `brain:ctx:{speaker_id}`)
- RAG via Qdrant para contexto semântico relevante

---

## 🔌 Interface NATS

### Entrada (request/reply)

```
Subject: mordomo.brain.generate

Payload:
{
  "speaker_id": "user_1",
  "text":       "acende a luz da sala",
  "confidence": 0.97
}

Reply:
{
  "text":        "Pronto, luz da sala acesa!",
  "tier":        "mordomo-simple",
  "intents":     ["iot_control"],
  "actions":     [{"type":"iot_control","device_id":"luz_sala","command":"turn_on"}],
  "tokens_used": 312
}
```

### Saída (publish)

```
Subject: mordomo.brain.action.{action_type}

Exemplo: mordomo.brain.action.iot_control
Payload:
{
  "speaker_id": "user_1",
  "device_id":  "luz_sala",
  "command":    "turn_on"
}
```

---

## 🔁 Arquitetura em 2 Fases

### Fase 1 — Classificação (Groq direct, ~200ms)

Chama `llama-3.3-70b-versatile` diretamente via API Groq para:
- Determinar o **tier** de complexidade da resposta
- Extrair **intents** da mensagem

```python
# src/llm.py
class ClassifyResult(TypedDict):
    tier:    str        # "mordomo-simple" | "mordomo-complex" | "mordomo-brain" | "mordomo-stakes"
    intents: list[str]  # ["iot_control", "pix_send", ...]
```

**Tiers e modelos correspondentes** (configurados no LiteLLM):

| Tier | Uso | Modelo |
|---|---|---|
| `mordomo-simple` | Comandos simples, IoT, lembretes | Gemini 3 Flash Preview |
| `mordomo-complex` | Raciocínio multi-etapa, contexto longo | Claude Haiku 4.5 |
| `mordomo-brain` | Respostas elaboradas, casa inteligente | Claude Sonnet 4.6 |
| `mordomo-stakes` | Ações com efeito real (PIX, alarmes) | Gemini 3 Pro Preview |

### Fase 2 — Geração com Function Calling (LiteLLM gateway)

Usa o tier determinado na Fase 1 como `model` no LiteLLM, com:
- Ferramentas carregadas do Redis (`mordomo:tools`)
- Histórico de conversa do Redis (`brain:ctx:{speaker_id}`)
- Contexto semântico do Qdrant (RAG, top-5 relevantes)

```python
# src/llm.py
async def generate(text, speaker_id, tier, tools) -> GenerateResult
```

---

## 🔧 Ferramentas Dinâmicas (Redis)

**Chave Redis:** `mordomo:tools` (HSET — campo = nome da ferramenta, valor = JSON do schema)

**Seed inicial** (7 ferramentas, via HSETNX):

| Ferramenta | Ação |
|---|---|
| `iot_control` | Controlar dispositivos IoT (liga/desliga/dimmer/temperatura) |
| `alarm_control` | Armar/desarmar alarme de segurança |
| `media_control` | Controlar TV/streaming (ligar, pausar, volume) |
| `pix_send` | Transferência PIX |
| `balance_query` | Consultar saldo bancário |
| `reminder_create` | Criar lembrete com data/hora |
| `openclaw_execute` | Enviar comando para agente OpenClaw |

```bash
# Adicionar nova ferramenta em runtime (sem restart)
redis-cli -n 1 HSET mordomo:tools nova_ferramenta '{"name":"nova_ferramenta","description":"...","parameters":{}}'
```

---

## 🎚️ Tiers Dinâmicos

Os tiers são descobertos via `GET /models` do LiteLLM gateway, filtrados pelo prefixo `mordomo-`. Cache TTL 300s.

Se o tier retornado pela Fase 1 não existir mais no LiteLLM, usa `TIER_FALLBACK = "mordomo-simple"`.

---

## ⚙️ Configuração (Variáveis de Ambiente)

| Variável | Default | Descrição |
|---|---|---|
| `NATS_URL` | `nats://nats:4222` | Servidor NATS |
| `REDIS_URL` | `redis://redis:6379/1` | Redis db1 |
| `QDRANT_URL` | `http://qdrant:6333` | Qdrant RAG |
| `LITELLM_URL` | `http://llm-gateway:4000` | LiteLLM gateway |
| `LITELLM_MASTER_KEY` | — | API key do LiteLLM |
| `GROQ_API_KEY` | — | API key Groq (Fase 1) |
| `TIER_PREFIX` | `mordomo-` | Prefixo para filtrar tiers no LiteLLM |
| `TIER_FALLBACK` | `mordomo-simple` | Tier usado se o escolhido indisponível |
| `TIER_CACHE_TTL` | `300` | TTL (segundos) do cache de tiers |
| `TOOLS_CACHE_TTL` | `120` | TTL (segundos) do cache de ferramentas |

---

## 🗂️ Estrutura de Arquivos

```
src/
  config.py        # Variáveis de ambiente e constantes
  llm.py           # Cliente LLM — classify() + generate()
  tools.py         # Carrega mordomo:tools do Redis com cache
  tools_seed.py    # Seed inicial das 7 ferramentas (HSETNX)
  tiers.py         # Descobre tiers via LiteLLM /models com cache
  actions.py       # extract_actions(llm_response) -> (text, tool_calls)
  handlers.py      # handle_generate(): orquestra phases 1+2, publica ações
  main.py          # Conecta NATS, init_tiers(), init_tools(), subscribe
```

---

## 💾 Uso do Redis (db1)

| Chave | Tipo | TTL | Conteúdo |
|---|---|---|---|
| `brain:ctx:{speaker_id}` | String (JSON) | `CTX_TTL` | Histórico de conversa (messages[]) |
| `mordomo:tools` | Hash | 120s | Schemas das ferramentas disponíveis |
| `mordomo:routes` | Hash | 120s | Rotas de ação (usado pelo orchestrator) |

---

## 🔄 Fluxo Completo

```
[orchestrator] mordomo.brain.generate (req/reply)
       ↓
[brain] Fase 1: Groq classify → {tier, intents}
       ↓
[brain] Fase 2: LiteLLM generate (tier, tools, ctx, RAG)
       ↓
[brain] extract_actions → (response_text, tool_calls[])
       ↓
[brain] publish mordomo.brain.action.{type} para cada ação
       ↓
[brain] reply {text, tier, intents, actions, tokens_used}
```

---

## 🚀 CI/CD

Build automático via GitHub Actions → `ghcr.io/aslamsys/mordomo-brain:latest`

Workflow: [`.github/workflows/ci.yml`](.github/workflows/ci.yml) — usa reusable workflow `AslamSys/.github`