# 🧠 Mordomo Brain (LLM)

## 🔗 Navegação

**[🏠 AslamSys](https://github.com/AslamSys)** → **[📚 _system](https://github.com/AslamSys/_system)** → **[📂 Aslam (Orange Pi 5 16GB)](https://github.com/AslamSys/_system/blob/main/hardware/mordomo%20-%20(orange-pi-5-16gb)/README.md)** → **mordomo-brain**

### Containers Relacionados (aslam)
- [mordomo-audio-bridge](https://github.com/AslamSys/mordomo-audio-bridge)
- [mordomo-audio-capture-vad](https://github.com/AslamSys/mordomo-audio-capture-vad)
- [mordomo-wake-word-detector](https://github.com/AslamSys/mordomo-wake-word-detector)
- [mordomo-speaker-verification](https://github.com/AslamSys/mordomo-speaker-verification)
- [mordomo-whisper-asr](https://github.com/AslamSys/mordomo-whisper-asr)
- [mordomo-speaker-id-diarization](https://github.com/AslamSys/mordomo-speaker-id-diarization)
- [mordomo-source-separation](https://github.com/AslamSys/mordomo-source-separation)
- [mordomo-core-gateway](https://github.com/AslamSys/mordomo-core-gateway)
- [mordomo-orchestrator](https://github.com/AslamSys/mordomo-orchestrator)
- [mordomo-tts-engine](https://github.com/AslamSys/mordomo-tts-engine)
- [mordomo-system-watchdog](https://github.com/AslamSys/mordomo-system-watchdog)
- [mordomo-dashboard-ui](https://github.com/AslamSys/mordomo-dashboard-ui)
- [mordomo-openclaw-agent](https://github.com/AslamSys/mordomo-openclaw-agent)
- [mordomo-action-dispatcher](https://github.com/AslamSys/mordomo-action-dispatcher)
- [mordomo-skills-runner](https://github.com/AslamSys/mordomo-skills-runner)

---

**Container:** `mordomo-brain`  
**Ecossistema:** Mordomo  
**Posição no Fluxo:** Sexto - Inteligência e Raciocínio

---

## 📋 Propósito

Cérebro do assistente: processa linguagem natural, mantém contexto por usuário, executa raciocínio, toma decisões e gera respostas contextualizadas. Opera em modo local-first com fallback para APIs cloud.

---

## 🎯 Responsabilidades

### Primárias
- ✅ Processar texto transcrito e gerar respostas naturais
- ✅ Manter contexto individualizado por speaker_id
- ✅ Detectar intenções (IoT, consultas, lembretes, etc)
- ✅ Executar raciocínio multi-step quando necessário
- ✅ Gerenciar estratégias de LLM (local-first, cloud-only, mixed)
- ✅ Integrar com Qdrant para RAG (busca semântica)
- ✅ Invocar ações (controlar dispositivos, criar lembretes)

### Secundárias
- ✅ Cache de respostas frequentes
- ✅ Summarização de contexto longo
- ✅ Detecção de mudança de tópico
- ✅ Fallback automático local ↔ cloud
- ✅ Token counting e otimização

---

## 🔧 Tecnologias

### LLM Cloud (Padrão)

O Orange Pi 5 **não roda LLM local**. Toda inferência é feita via API cloud.

```yaml
OpenAI: gpt-4o-mini (padrão), gpt-4o (complexo)
Anthropic: claude-3.5-haiku (padrão), claude-3.5-sonnet (complexo)
Google: gemini-2.0-flash (padrão), gemini-2.0-pro (complexo)
Groq: llama-3.3-70b (baixa latência, fallback gratuito)
```

### LLM Local (Futuro — Jetson Orin dedicado)

> ⚠️ **Não implementado no Orange Pi 5.** Quando o Jetson Orin for adquirido, um serviço de inferência local (Ollama ou TensorRT-LLM) será exposto via API compatível com OpenAI na rede interna. O brain apenas aponta o endpoint — sem Ollama no Orange Pi.

```yaml
Futuro endpoint local: http://jetson-orin:11434/v1
Modelo candidato: qwen2.5:7b (Jetson tem VRAM dedicada)
```

### Stack Adicional
```python
langchain  # Chains e agents
qdrant-client  # RAG
tiktoken  # Token counting
sentence-transformers  # Embeddings
```

---

## 📊 Especificações

```yaml
Hardware (Orange Pi 5 Ultra):
  RAM do brain: ~200 MB  # apenas o processo Python + LangChain + cliente HTTP
  CPU: < 5% idle (sem inferência local)
  Sem GPU dedicada para LLM

Performance Cloud (padrão):
  gpt-4o-mini:       latência 500-1500ms | $0.15/1M tokens input
  claude-3.5-haiku:  latência 400-1200ms | $0.80/1M tokens input
  gemini-2.0-flash:  latência 300-900ms  | $0.075/1M tokens input  ← mais barato
  groq llama-3.3-70b: latência 200-600ms | gratuito (rate limited)  ← fallback

Estrategias:
  cloud-primary:  100% cloud, modelo escolhido por complexidade da query
  groq-fallback:  se APIs pagas indisponíveis, usa Groq gratuito
  # local-inference: FUTURO — requer Jetson Orin na rede
```

---

## 🔌 Interfaces

### Input (gRPC)
```protobuf
service BrainService {
  rpc Generate(GenerateRequest) returns (GenerateResponse);
  rpc GenerateStream(GenerateRequest) returns (stream GenerateStreamResponse);
}

message GenerateRequest {
  string text = 1;
  string speaker_id = 2;
  repeated Message conversation_history = 3;
  repeated ContextItem context = 4;  // Do Qdrant
  string strategy = 5;  // "local-first" | "cloud-only" | "mixed"
  map<string, string> metadata = 6;
}

message Message {
  string role = 1;  // "user" | "assistant"
  string content = 2;
  int64 timestamp = 3;
}

message ContextItem {
  string text = 1;
  float score = 2;  // Relevância
}

message GenerateResponse {
  string text = 1;
  float confidence = 2;
  repeated Action actions = 3;
  string model_used = 4;
  int32 tokens_used = 5;
}

message Action {
  string type = 1;  // "iot_control" | "reminder" | "query"
  string target = 2;
  map<string, string> params = 3;
}
```

### Output (NATS)
```python
# Ações detectadas

## 🔗 Navegação

**[🏠 AslamSys](https://github.com/AslamSys)** → **[📚 _system](https://github.com/AslamSys/_system)** → **[📂 Aslam (Orange Pi 5 16GB)](https://github.com/AslamSys/_system/blob/main/hardware/mordomo%20-%20(orange-pi-5-16gb)/README.md)** → **mordomo-brain**

### Containers Relacionados (aslam)
- [mordomo-audio-bridge](https://github.com/AslamSys/mordomo-audio-bridge)
- [mordomo-audio-capture-vad](https://github.com/AslamSys/mordomo-audio-capture-vad)
- [mordomo-wake-word-detector](https://github.com/AslamSys/mordomo-wake-word-detector)
- [mordomo-speaker-verification](https://github.com/AslamSys/mordomo-speaker-verification)
- [mordomo-whisper-asr](https://github.com/AslamSys/mordomo-whisper-asr)
- [mordomo-speaker-id-diarization](https://github.com/AslamSys/mordomo-speaker-id-diarization)
- [mordomo-source-separation](https://github.com/AslamSys/mordomo-source-separation)
- [mordomo-core-gateway](https://github.com/AslamSys/mordomo-core-gateway)
- [mordomo-orchestrator](https://github.com/AslamSys/mordomo-orchestrator)
- [mordomo-tts-engine](https://github.com/AslamSys/mordomo-tts-engine)
- [mordomo-system-watchdog](https://github.com/AslamSys/mordomo-system-watchdog)
- [mordomo-dashboard-ui](https://github.com/AslamSys/mordomo-dashboard-ui)
- [mordomo-openclaw-agent](https://github.com/AslamSys/mordomo-openclaw-agent)
- [mordomo-action-dispatcher](https://github.com/AslamSys/mordomo-action-dispatcher)
- [mordomo-skills-runner](https://github.com/AslamSys/mordomo-skills-runner)

---
subject: "brain.action.{action_type}"
payload: {
  "action": "iot_control",
  "device_id": "light_sala",
  "command": "turn_on",
  "speaker_id": "user_1",
  "timestamp": 1732723200.123
}
```

---

## ⚙️ Configuração

```yaml
ollama:
  endpoint: "http://localhost:11434"
  model: "qwen2.5:3b"
  temperature: 0.7
  top_p: 0.9
  top_k: 40
  repeat_penalty: 1.1
  num_ctx: 8192  # Context window
  num_predict: 512  # Max tokens
  
cloud_apis:
  openai:
    api_key: "${OPENAI_API_KEY}"
    model: "gpt-4o-mini"
    max_tokens: 500
    
  anthropic:
    api_key: "${ANTHROPIC_API_KEY}"
    model: "claude-3-haiku-20240307"
    
  groq:
    api_key: "${GROQ_API_KEY}"
    model: "llama-3.3-70b-versatile"

strategy:
  default: "cloud-primary"

  # Roteamento por complexidade
  routing:
    simple_command:    gemini-2.0-flash   # IoT, lembretes, perguntas curtas
    complex_reasoning: gpt-4o-mini        # multi-step, finanças, contexto longo
    high_stakes:       claude-3.5-haiku   # ações com efeito real (PIX, alarmes)

  # Fallback em cascata (sem local)
  fallback_chain:
    - gemini-2.0-flash
    - gpt-4o-mini
    - groq/llama-3.3-70b    # gratuito, último recurso

  # Local inference: DESABILITADO no Orange Pi
  local_inference:
    enabled: false
    # Habilitar quando Jetson Orin estiver na rede:
    # endpoint: http://jetson-orin:11434/v1
    # model: qwen2.5:7b

rag:
  enabled: true
  qdrant_url: "http://qdrant:6333"
  collection: "conversations"
  top_k: 5
  min_score: 0.7
  
context:
  max_messages: 10
  summarize_after: 20
  context_window_tokens: 6000

cache:
  enabled: true
  ttl: 3600  # segundos
  max_entries: 1000

prompts:
  system: |
    Você é Aslam, um assistente doméstico inteligente.
    Você controla dispositivos IoT, responde perguntas e ajuda os moradores.
    Seja conciso, útil e natural nas respostas.
    
    Usuários autorizados:
    - user_1: Dono da casa
    - user_2: Esposa
    
    Dispositivos disponíveis:
    - Luzes: sala, quarto, cozinha
    - Ar condicionado: sala, quarto
    - Persianas: sala
    
  user_template: |
    {history}
    
    Contexto relevante:
    {context}
    
    Usuário ({speaker_id}): {text}
    
    Responda de forma natural e execute ações se necessário.
```

---

## 🎯 Detecção de Intenções

```python
# Intent Classification

## 🔗 Navegação

**[🏠 AslamSys](https://github.com/AslamSys)** → **[📚 _system](https://github.com/AslamSys/_system)** → **[📂 Aslam (Orange Pi 5 16GB)](https://github.com/AslamSys/_system/blob/main/hardware/mordomo%20-%20(orange-pi-5-16gb)/README.md)** → **mordomo-brain**

### Containers Relacionados (aslam)
- [mordomo-audio-bridge](https://github.com/AslamSys/mordomo-audio-bridge)
- [mordomo-audio-capture-vad](https://github.com/AslamSys/mordomo-audio-capture-vad)
- [mordomo-wake-word-detector](https://github.com/AslamSys/mordomo-wake-word-detector)
- [mordomo-speaker-verification](https://github.com/AslamSys/mordomo-speaker-verification)
- [mordomo-whisper-asr](https://github.com/AslamSys/mordomo-whisper-asr)
- [mordomo-speaker-id-diarization](https://github.com/AslamSys/mordomo-speaker-id-diarization)
- [mordomo-source-separation](https://github.com/AslamSys/mordomo-source-separation)
- [mordomo-core-gateway](https://github.com/AslamSys/mordomo-core-gateway)
- [mordomo-orchestrator](https://github.com/AslamSys/mordomo-orchestrator)
- [mordomo-tts-engine](https://github.com/AslamSys/mordomo-tts-engine)
- [mordomo-system-watchdog](https://github.com/AslamSys/mordomo-system-watchdog)
- [mordomo-dashboard-ui](https://github.com/AslamSys/mordomo-dashboard-ui)
- [mordomo-openclaw-agent](https://github.com/AslamSys/mordomo-openclaw-agent)
- [mordomo-action-dispatcher](https://github.com/AslamSys/mordomo-action-dispatcher)
- [mordomo-skills-runner](https://github.com/AslamSys/mordomo-skills-runner)

---
intents = {
  "iot_control": {
    "keywords": ["acende", "apaga", "aumenta", "diminui", "abre", "fecha"],
    "action_type": "iot_control"
  },
  
  "query_weather": {
    "keywords": ["temperatura", "clima", "tempo", "previsão"],
    "action_type": "query",
    "requires_cloud": true
  },
  
  "query_time": {
    "keywords": ["que horas", "horário"],
    "action_type": "query"
  },
  
  "reminder": {
    "keywords": ["lembrar", "lembre-me", "criar lembrete"],
    "action_type": "reminder"
  },
  
  "general_conversation": {
    "default": true
  }
}

# Exemplo de detecção

## 🔗 Navegação

**[🏠 AslamSys](https://github.com/AslamSys)** → **[📚 _system](https://github.com/AslamSys/_system)** → **[📂 Aslam (Orange Pi 5 16GB)](https://github.com/AslamSys/_system/blob/main/hardware/mordomo%20-%20(orange-pi-5-16gb)/README.md)** → **mordomo-brain**

### Containers Relacionados (aslam)
- [mordomo-audio-bridge](https://github.com/AslamSys/mordomo-audio-bridge)
- [mordomo-audio-capture-vad](https://github.com/AslamSys/mordomo-audio-capture-vad)
- [mordomo-wake-word-detector](https://github.com/AslamSys/mordomo-wake-word-detector)
- [mordomo-speaker-verification](https://github.com/AslamSys/mordomo-speaker-verification)
- [mordomo-whisper-asr](https://github.com/AslamSys/mordomo-whisper-asr)
- [mordomo-speaker-id-diarization](https://github.com/AslamSys/mordomo-speaker-id-diarization)
- [mordomo-source-separation](https://github.com/AslamSys/mordomo-source-separation)
- [mordomo-core-gateway](https://github.com/AslamSys/mordomo-core-gateway)
- [mordomo-orchestrator](https://github.com/AslamSys/mordomo-orchestrator)
- [mordomo-tts-engine](https://github.com/AslamSys/mordomo-tts-engine)
- [mordomo-system-watchdog](https://github.com/AslamSys/mordomo-system-watchdog)
- [mordomo-dashboard-ui](https://github.com/AslamSys/mordomo-dashboard-ui)
- [mordomo-openclaw-agent](https://github.com/AslamSys/mordomo-openclaw-agent)
- [mordomo-action-dispatcher](https://github.com/AslamSys/mordomo-action-dispatcher)
- [mordomo-skills-runner](https://github.com/AslamSys/mordomo-skills-runner)

---
user_input = "acende a luz da sala"
intent = detect_intent(user_input)  # "iot_control"
entities = extract_entities(user_input)  # {"device": "light_sala", "action": "turn_on"}
```

---

## 🔄 Fluxo de Processamento

```python
async def generate_response(request: GenerateRequest):
    # 1. Carregar contexto do histórico
    history = build_conversation_history(
        request.conversation_history,
        max_messages=10
    )
    
    # 2. Buscar contexto semântico (RAG)
    if rag_enabled:
        embedding = await get_embedding(request.text)
        context = await qdrant.search(
            collection="conversations",
            query_vector=embedding,
            limit=5,
            filter={"speaker_id": request.speaker_id}
        )
    
    # 3. Detectar intenção
    intent = detect_intent(request.text)
    entities = extract_entities(request.text, intent)
    
    # 4. Selecionar estratégia LLM
    strategy = select_strategy(request.strategy, intent)
    
    # 5. Gerar resposta
    if strategy == "local-first":
        try:
            response = await call_ollama(request, history, context)
        except Exception as e:
            logger.warning(f"Local LLM failed: {e}, falling back to cloud")
            response = await call_cloud_llm(request, history, context)
    
    elif strategy == "cloud-only":
        response = await call_cloud_llm(request, history, context)
    
    elif strategy == "mixed":
        # Local reasoning + Cloud refinement
        local_response = await call_ollama(request, history, context)
        if local_response.confidence < 0.7:
            response = await call_cloud_llm(request, history, context)
        else:
            response = local_response
    
    # 6. Executar ações detectadas
    actions = []
    if intent in ["iot_control", "reminder"]:
        action = create_action(intent, entities)
        await execute_action(action)
        actions.append(action)
    
    # 7. Cachear resposta
    if cache_enabled:
        await cache.set(
            key=hash(request.text + request.speaker_id),
            value=response,
            ttl=3600
        )
    
    # 8. Retornar
    return GenerateResponse(
        text=response.text,
        confidence=response.confidence,
        actions=actions,
        model_used=strategy,
        tokens_used=response.tokens
    )
```

---

## 📈 Métricas

```python
# Uso de LLM

## 🔗 Navegação

**[🏠 AslamSys](https://github.com/AslamSys)** → **[📚 _system](https://github.com/AslamSys/_system)** → **[📂 Aslam (Orange Pi 5 16GB)](https://github.com/AslamSys/_system/blob/main/hardware/mordomo%20-%20(orange-pi-5-16gb)/README.md)** → **mordomo-brain**

### Containers Relacionados (aslam)
- [mordomo-audio-bridge](https://github.com/AslamSys/mordomo-audio-bridge)
- [mordomo-audio-capture-vad](https://github.com/AslamSys/mordomo-audio-capture-vad)
- [mordomo-wake-word-detector](https://github.com/AslamSys/mordomo-wake-word-detector)
- [mordomo-speaker-verification](https://github.com/AslamSys/mordomo-speaker-verification)
- [mordomo-whisper-asr](https://github.com/AslamSys/mordomo-whisper-asr)
- [mordomo-speaker-id-diarization](https://github.com/AslamSys/mordomo-speaker-id-diarization)
- [mordomo-source-separation](https://github.com/AslamSys/mordomo-source-separation)
- [mordomo-core-gateway](https://github.com/AslamSys/mordomo-core-gateway)
- [mordomo-orchestrator](https://github.com/AslamSys/mordomo-orchestrator)
- [mordomo-tts-engine](https://github.com/AslamSys/mordomo-tts-engine)
- [mordomo-system-watchdog](https://github.com/AslamSys/mordomo-system-watchdog)
- [mordomo-dashboard-ui](https://github.com/AslamSys/mordomo-dashboard-ui)
- [mordomo-openclaw-agent](https://github.com/AslamSys/mordomo-openclaw-agent)
- [mordomo-action-dispatcher](https://github.com/AslamSys/mordomo-action-dispatcher)
- [mordomo-skills-runner](https://github.com/AslamSys/mordomo-skills-runner)

---
llm_requests_total{model, strategy}
llm_tokens_total{model, type}  # type: input/output
llm_latency_seconds{model, percentile}
llm_cost_usd{model}

# Performance

## 🔗 Navegação

**[🏠 AslamSys](https://github.com/AslamSys)** → **[📚 _system](https://github.com/AslamSys/_system)** → **[📂 Aslam (Orange Pi 5 16GB)](https://github.com/AslamSys/_system/blob/main/hardware/mordomo%20-%20(orange-pi-5-16gb)/README.md)** → **mordomo-brain**

### Containers Relacionados (aslam)
- [mordomo-audio-bridge](https://github.com/AslamSys/mordomo-audio-bridge)
- [mordomo-audio-capture-vad](https://github.com/AslamSys/mordomo-audio-capture-vad)
- [mordomo-wake-word-detector](https://github.com/AslamSys/mordomo-wake-word-detector)
- [mordomo-speaker-verification](https://github.com/AslamSys/mordomo-speaker-verification)
- [mordomo-whisper-asr](https://github.com/AslamSys/mordomo-whisper-asr)
- [mordomo-speaker-id-diarization](https://github.com/AslamSys/mordomo-speaker-id-diarization)
- [mordomo-source-separation](https://github.com/AslamSys/mordomo-source-separation)
- [mordomo-core-gateway](https://github.com/AslamSys/mordomo-core-gateway)
- [mordomo-orchestrator](https://github.com/AslamSys/mordomo-orchestrator)
- [mordomo-tts-engine](https://github.com/AslamSys/mordomo-tts-engine)
- [mordomo-system-watchdog](https://github.com/AslamSys/mordomo-system-watchdog)
- [mordomo-dashboard-ui](https://github.com/AslamSys/mordomo-dashboard-ui)
- [mordomo-openclaw-agent](https://github.com/AslamSys/mordomo-openclaw-agent)
- [mordomo-action-dispatcher](https://github.com/AslamSys/mordomo-action-dispatcher)
- [mordomo-skills-runner](https://github.com/AslamSys/mordomo-skills-runner)

---
llm_cache_hits_total
llm_cache_misses_total
llm_fallback_total{from, to}

# Qualidade

## 🔗 Navegação

**[🏠 AslamSys](https://github.com/AslamSys)** → **[📚 _system](https://github.com/AslamSys/_system)** → **[📂 Aslam (Orange Pi 5 16GB)](https://github.com/AslamSys/_system/blob/main/hardware/mordomo%20-%20(orange-pi-5-16gb)/README.md)** → **mordomo-brain**

### Containers Relacionados (aslam)
- [mordomo-audio-bridge](https://github.com/AslamSys/mordomo-audio-bridge)
- [mordomo-audio-capture-vad](https://github.com/AslamSys/mordomo-audio-capture-vad)
- [mordomo-wake-word-detector](https://github.com/AslamSys/mordomo-wake-word-detector)
- [mordomo-speaker-verification](https://github.com/AslamSys/mordomo-speaker-verification)
- [mordomo-whisper-asr](https://github.com/AslamSys/mordomo-whisper-asr)
- [mordomo-speaker-id-diarization](https://github.com/AslamSys/mordomo-speaker-id-diarization)
- [mordomo-source-separation](https://github.com/AslamSys/mordomo-source-separation)
- [mordomo-core-gateway](https://github.com/AslamSys/mordomo-core-gateway)
- [mordomo-orchestrator](https://github.com/AslamSys/mordomo-orchestrator)
- [mordomo-tts-engine](https://github.com/AslamSys/mordomo-tts-engine)
- [mordomo-system-watchdog](https://github.com/AslamSys/mordomo-system-watchdog)
- [mordomo-dashboard-ui](https://github.com/AslamSys/mordomo-dashboard-ui)
- [mordomo-openclaw-agent](https://github.com/AslamSys/mordomo-openclaw-agent)
- [mordomo-action-dispatcher](https://github.com/AslamSys/mordomo-action-dispatcher)
- [mordomo-skills-runner](https://github.com/AslamSys/mordomo-skills-runner)

---
llm_confidence_avg{model}
llm_intent_accuracy

# Ações

## 🔗 Navegação

**[🏠 AslamSys](https://github.com/AslamSys)** → **[📚 _system](https://github.com/AslamSys/_system)** → **[📂 Aslam (Orange Pi 5 16GB)](https://github.com/AslamSys/_system/blob/main/hardware/mordomo%20-%20(orange-pi-5-16gb)/README.md)** → **mordomo-brain**

### Containers Relacionados (aslam)
- [mordomo-audio-bridge](https://github.com/AslamSys/mordomo-audio-bridge)
- [mordomo-audio-capture-vad](https://github.com/AslamSys/mordomo-audio-capture-vad)
- [mordomo-wake-word-detector](https://github.com/AslamSys/mordomo-wake-word-detector)
- [mordomo-speaker-verification](https://github.com/AslamSys/mordomo-speaker-verification)
- [mordomo-whisper-asr](https://github.com/AslamSys/mordomo-whisper-asr)
- [mordomo-speaker-id-diarization](https://github.com/AslamSys/mordomo-speaker-id-diarization)
- [mordomo-source-separation](https://github.com/AslamSys/mordomo-source-separation)
- [mordomo-core-gateway](https://github.com/AslamSys/mordomo-core-gateway)
- [mordomo-orchestrator](https://github.com/AslamSys/mordomo-orchestrator)
- [mordomo-tts-engine](https://github.com/AslamSys/mordomo-tts-engine)
- [mordomo-system-watchdog](https://github.com/AslamSys/mordomo-system-watchdog)
- [mordomo-dashboard-ui](https://github.com/AslamSys/mordomo-dashboard-ui)
- [mordomo-openclaw-agent](https://github.com/AslamSys/mordomo-openclaw-agent)
- [mordomo-action-dispatcher](https://github.com/AslamSys/mordomo-action-dispatcher)
- [mordomo-skills-runner](https://github.com/AslamSys/mordomo-skills-runner)

---
actions_executed_total{type}
actions_failed_total{type, reason}
```

---

## 🐳 Docker

```dockerfile
FROM python:3.11-slim

# Ollama (será instalado no host, não no container)

## 🔗 Navegação

**[🏠 AslamSys](https://github.com/AslamSys)** → **[📚 _system](https://github.com/AslamSys/_system)** → **[📂 Aslam (Orange Pi 5 16GB)](https://github.com/AslamSys/_system/blob/main/hardware/mordomo%20-%20(orange-pi-5-16gb)/README.md)** → **mordomo-brain**

### Containers Relacionados (aslam)
- [mordomo-audio-bridge](https://github.com/AslamSys/mordomo-audio-bridge)
- [mordomo-audio-capture-vad](https://github.com/AslamSys/mordomo-audio-capture-vad)
- [mordomo-wake-word-detector](https://github.com/AslamSys/mordomo-wake-word-detector)
- [mordomo-speaker-verification](https://github.com/AslamSys/mordomo-speaker-verification)
- [mordomo-whisper-asr](https://github.com/AslamSys/mordomo-whisper-asr)
- [mordomo-speaker-id-diarization](https://github.com/AslamSys/mordomo-speaker-id-diarization)
- [mordomo-source-separation](https://github.com/AslamSys/mordomo-source-separation)
- [mordomo-core-gateway](https://github.com/AslamSys/mordomo-core-gateway)
- [mordomo-orchestrator](https://github.com/AslamSys/mordomo-orchestrator)
- [mordomo-tts-engine](https://github.com/AslamSys/mordomo-tts-engine)
- [mordomo-system-watchdog](https://github.com/AslamSys/mordomo-system-watchdog)
- [mordomo-dashboard-ui](https://github.com/AslamSys/mordomo-dashboard-ui)
- [mordomo-openclaw-agent](https://github.com/AslamSys/mordomo-openclaw-agent)
- [mordomo-action-dispatcher](https://github.com/AslamSys/mordomo-action-dispatcher)
- [mordomo-skills-runner](https://github.com/AslamSys/mordomo-skills-runner)

---
# Este container apenas se conecta ao Ollama via HTTP

## 🔗 Navegação

**[🏠 AslamSys](https://github.com/AslamSys)** → **[📚 _system](https://github.com/AslamSys/_system)** → **[📂 Aslam (Orange Pi 5 16GB)](https://github.com/AslamSys/_system/blob/main/hardware/mordomo%20-%20(orange-pi-5-16gb)/README.md)** → **mordomo-brain**

### Containers Relacionados (aslam)
- [mordomo-audio-bridge](https://github.com/AslamSys/mordomo-audio-bridge)
- [mordomo-audio-capture-vad](https://github.com/AslamSys/mordomo-audio-capture-vad)
- [mordomo-wake-word-detector](https://github.com/AslamSys/mordomo-wake-word-detector)
- [mordomo-speaker-verification](https://github.com/AslamSys/mordomo-speaker-verification)
- [mordomo-whisper-asr](https://github.com/AslamSys/mordomo-whisper-asr)
- [mordomo-speaker-id-diarization](https://github.com/AslamSys/mordomo-speaker-id-diarization)
- [mordomo-source-separation](https://github.com/AslamSys/mordomo-source-separation)
- [mordomo-core-gateway](https://github.com/AslamSys/mordomo-core-gateway)
- [mordomo-orchestrator](https://github.com/AslamSys/mordomo-orchestrator)
- [mordomo-tts-engine](https://github.com/AslamSys/mordomo-tts-engine)
- [mordomo-system-watchdog](https://github.com/AslamSys/mordomo-system-watchdog)
- [mordomo-dashboard-ui](https://github.com/AslamSys/mordomo-dashboard-ui)
- [mordomo-openclaw-agent](https://github.com/AslamSys/mordomo-openclaw-agent)
- [mordomo-action-dispatcher](https://github.com/AslamSys/mordomo-action-dispatcher)
- [mordomo-skills-runner](https://github.com/AslamSys/mordomo-skills-runner)

---

WORKDIR /app

# Dependencies

## 🔗 Navegação

**[🏠 AslamSys](https://github.com/AslamSys)** → **[📚 _system](https://github.com/AslamSys/_system)** → **[📂 Aslam (Orange Pi 5 16GB)](https://github.com/AslamSys/_system/blob/main/hardware/mordomo%20-%20(orange-pi-5-16gb)/README.md)** → **mordomo-brain**

### Containers Relacionados (aslam)
- [mordomo-audio-bridge](https://github.com/AslamSys/mordomo-audio-bridge)
- [mordomo-audio-capture-vad](https://github.com/AslamSys/mordomo-audio-capture-vad)
- [mordomo-wake-word-detector](https://github.com/AslamSys/mordomo-wake-word-detector)
- [mordomo-speaker-verification](https://github.com/AslamSys/mordomo-speaker-verification)
- [mordomo-whisper-asr](https://github.com/AslamSys/mordomo-whisper-asr)
- [mordomo-speaker-id-diarization](https://github.com/AslamSys/mordomo-speaker-id-diarization)
- [mordomo-source-separation](https://github.com/AslamSys/mordomo-source-separation)
- [mordomo-core-gateway](https://github.com/AslamSys/mordomo-core-gateway)
- [mordomo-orchestrator](https://github.com/AslamSys/mordomo-orchestrator)
- [mordomo-tts-engine](https://github.com/AslamSys/mordomo-tts-engine)
- [mordomo-system-watchdog](https://github.com/AslamSys/mordomo-system-watchdog)
- [mordomo-dashboard-ui](https://github.com/AslamSys/mordomo-dashboard-ui)
- [mordomo-openclaw-agent](https://github.com/AslamSys/mordomo-openclaw-agent)
- [mordomo-action-dispatcher](https://github.com/AslamSys/mordomo-action-dispatcher)
- [mordomo-skills-runner](https://github.com/AslamSys/mordomo-skills-runner)

---
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download embedding model

## 🔗 Navegação

**[🏠 AslamSys](https://github.com/AslamSys)** → **[📚 _system](https://github.com/AslamSys/_system)** → **[📂 Aslam (Orange Pi 5 16GB)](https://github.com/AslamSys/_system/blob/main/hardware/mordomo%20-%20(orange-pi-5-16gb)/README.md)** → **mordomo-brain**

### Containers Relacionados (aslam)
- [mordomo-audio-bridge](https://github.com/AslamSys/mordomo-audio-bridge)
- [mordomo-audio-capture-vad](https://github.com/AslamSys/mordomo-audio-capture-vad)
- [mordomo-wake-word-detector](https://github.com/AslamSys/mordomo-wake-word-detector)
- [mordomo-speaker-verification](https://github.com/AslamSys/mordomo-speaker-verification)
- [mordomo-whisper-asr](https://github.com/AslamSys/mordomo-whisper-asr)
- [mordomo-speaker-id-diarization](https://github.com/AslamSys/mordomo-speaker-id-diarization)
- [mordomo-source-separation](https://github.com/AslamSys/mordomo-source-separation)
- [mordomo-core-gateway](https://github.com/AslamSys/mordomo-core-gateway)
- [mordomo-orchestrator](https://github.com/AslamSys/mordomo-orchestrator)
- [mordomo-tts-engine](https://github.com/AslamSys/mordomo-tts-engine)
- [mordomo-system-watchdog](https://github.com/AslamSys/mordomo-system-watchdog)
- [mordomo-dashboard-ui](https://github.com/AslamSys/mordomo-dashboard-ui)
- [mordomo-openclaw-agent](https://github.com/AslamSys/mordomo-openclaw-agent)
- [mordomo-action-dispatcher](https://github.com/AslamSys/mordomo-action-dispatcher)
- [mordomo-skills-runner](https://github.com/AslamSys/mordomo-skills-runner)

---
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')"

# Application

## 🔗 Navegação

**[🏠 AslamSys](https://github.com/AslamSys)** → **[📚 _system](https://github.com/AslamSys/_system)** → **[📂 Aslam (Orange Pi 5 16GB)](https://github.com/AslamSys/_system/blob/main/hardware/mordomo%20-%20(orange-pi-5-16gb)/README.md)** → **mordomo-brain**

### Containers Relacionados (aslam)
- [mordomo-audio-bridge](https://github.com/AslamSys/mordomo-audio-bridge)
- [mordomo-audio-capture-vad](https://github.com/AslamSys/mordomo-audio-capture-vad)
- [mordomo-wake-word-detector](https://github.com/AslamSys/mordomo-wake-word-detector)
- [mordomo-speaker-verification](https://github.com/AslamSys/mordomo-speaker-verification)
- [mordomo-whisper-asr](https://github.com/AslamSys/mordomo-whisper-asr)
- [mordomo-speaker-id-diarization](https://github.com/AslamSys/mordomo-speaker-id-diarization)
- [mordomo-source-separation](https://github.com/AslamSys/mordomo-source-separation)
- [mordomo-core-gateway](https://github.com/AslamSys/mordomo-core-gateway)
- [mordomo-orchestrator](https://github.com/AslamSys/mordomo-orchestrator)
- [mordomo-tts-engine](https://github.com/AslamSys/mordomo-tts-engine)
- [mordomo-system-watchdog](https://github.com/AslamSys/mordomo-system-watchdog)
- [mordomo-dashboard-ui](https://github.com/AslamSys/mordomo-dashboard-ui)
- [mordomo-openclaw-agent](https://github.com/AslamSys/mordomo-openclaw-agent)
- [mordomo-action-dispatcher](https://github.com/AslamSys/mordomo-action-dispatcher)
- [mordomo-skills-runner](https://github.com/AslamSys/mordomo-skills-runner)

---
COPY src/ ./src/
COPY config/ ./config/
COPY prompts/ ./prompts/

EXPOSE 50052 8004

CMD ["python", "src/server.py"]
```

---

## 🧪 Testes

```python
# test_brain.py

## 🔗 Navegação

**[🏠 AslamSys](https://github.com/AslamSys)** → **[📚 _system](https://github.com/AslamSys/_system)** → **[📂 Aslam (Orange Pi 5 16GB)](https://github.com/AslamSys/_system/blob/main/hardware/mordomo%20-%20(orange-pi-5-16gb)/README.md)** → **mordomo-brain**

### Containers Relacionados (aslam)
- [mordomo-audio-bridge](https://github.com/AslamSys/mordomo-audio-bridge)
- [mordomo-audio-capture-vad](https://github.com/AslamSys/mordomo-audio-capture-vad)
- [mordomo-wake-word-detector](https://github.com/AslamSys/mordomo-wake-word-detector)
- [mordomo-speaker-verification](https://github.com/AslamSys/mordomo-speaker-verification)
- [mordomo-whisper-asr](https://github.com/AslamSys/mordomo-whisper-asr)
- [mordomo-speaker-id-diarization](https://github.com/AslamSys/mordomo-speaker-id-diarization)
- [mordomo-source-separation](https://github.com/AslamSys/mordomo-source-separation)
- [mordomo-core-gateway](https://github.com/AslamSys/mordomo-core-gateway)
- [mordomo-orchestrator](https://github.com/AslamSys/mordomo-orchestrator)
- [mordomo-tts-engine](https://github.com/AslamSys/mordomo-tts-engine)
- [mordomo-system-watchdog](https://github.com/AslamSys/mordomo-system-watchdog)
- [mordomo-dashboard-ui](https://github.com/AslamSys/mordomo-dashboard-ui)
- [mordomo-openclaw-agent](https://github.com/AslamSys/mordomo-openclaw-agent)
- [mordomo-action-dispatcher](https://github.com/AslamSys/mordomo-action-dispatcher)
- [mordomo-skills-runner](https://github.com/AslamSys/mordomo-skills-runner)

---
def test_local_llm_generation():
    request = GenerateRequest(
        text="qual a temperatura",
        speaker_id="user_1",
        strategy="local-first"
    )
    response = brain.generate(request)
    
    assert response.text is not None
    assert response.model_used == "qwen2.5:3b"
    assert response.tokens_used > 0

def test_intent_detection():
    intents = [
        ("acende a luz da sala", "iot_control"),
        ("que horas são", "query_time"),
        ("qual a previsão do tempo", "query_weather"),
    ]
    
    for text, expected_intent in intents:
        detected = detect_intent(text)
        assert detected == expected_intent

def test_fallback_to_cloud():
    # Simula falha do Ollama
    with mock.patch('ollama.generate', side_effect=TimeoutError):
        request = GenerateRequest(
            text="teste",
            speaker_id="user_1",
            strategy="local-first"
        )
        response = brain.generate(request)
        
        assert "gpt" in response.model_used.lower() or "claude" in response.model_used.lower()

def test_rag_integration():
    # Testa busca no Qdrant
    request = GenerateRequest(
        text="o que eu perguntei ontem sobre temperatura",
        speaker_id="user_1"
    )
    
    response = brain.generate(request)
    assert "temperatura" in response.text.lower()
```

---

## 💡 Prompts Engineering

### System Prompt (Base)
```
Você é Aslam, assistente doméstico inteligente da casa.

PERSONALIDADE:
- Conciso e direto
- Prestativo e proativo
- Natural e conversacional
- Evita explicações longas

CAPACIDADES:
- Controlar dispositivos IoT (luzes, AC, persianas)
- Responder perguntas gerais
- Criar lembretes e tarefas
- Consultar clima, horário, etc.

USUÁRIOS:
- user_1: Dono (você)
- user_2: Esposa

REGRAS:
- Sempre confirme ações de IoT
- Use português brasileiro
- Seja contextual com base no histórico
- Execute ações quando solicitado
```

### Few-Shot Examples
```yaml
examples:
  - user: "acende a luz da sala"
    assistant: "Luz da sala acesa."
    action: {type: "iot_control", device: "light_sala", command: "turn_on"}
  
  - user: "qual a temperatura lá fora"
    assistant: "A temperatura atual é 28°C."
    action: {type: "query", source: "weather_api"}
  
  - user: "me lembre de ligar para o médico amanhã às 14h"
    assistant: "Lembrete criado para amanhã às 14h: ligar para o médico."
    action: {type: "reminder", datetime: "tomorrow_14:00", text: "ligar para o médico"}
```

---

## 🔗 Integração

**Recebe de:** Core API (gRPC)  
**Envia para:** 
- Core API (gRPC response)
- NATS (message broker da Infraestrutura) (ações detectadas)
- Qdrant (busca RAG)

**Dependências:**
- Ollama (local LLM server)
- Cloud APIs (fallback)
- Qdrant (contexto semântico)

**Monitora:** Prometheus, Loki

---

## 🚀 Deploy

### Docker Compose
```yaml
mordomo-brain:
  build: ./containers/mordomo-brain
  container_name: mordomo-brain
  environment:
    - OLLAMA_HOST=http://host.docker.internal:11434
    - OPENAI_API_KEY=${OPENAI_API_KEY}
    - QDRANT_URL=http://qdrant:6333
    - NATS_URL=nats://nats:4222
  ports:
    - "50052:50052"  # gRPC
    - "8004:8004"    # Metrics
  networks:
    - mordomo-net
  restart: unless-stopped
  
# Ollama (no host)

## 🔗 Navegação

**[🏠 AslamSys](https://github.com/AslamSys)** → **[📚 _system](https://github.com/AslamSys/_system)** → **[📂 Aslam (Orange Pi 5 16GB)](https://github.com/AslamSys/_system/blob/main/hardware/mordomo%20-%20(orange-pi-5-16gb)/README.md)** → **mordomo-brain**

### Containers Relacionados (aslam)
- [mordomo-audio-bridge](https://github.com/AslamSys/mordomo-audio-bridge)
- [mordomo-audio-capture-vad](https://github.com/AslamSys/mordomo-audio-capture-vad)
- [mordomo-wake-word-detector](https://github.com/AslamSys/mordomo-wake-word-detector)
- [mordomo-speaker-verification](https://github.com/AslamSys/mordomo-speaker-verification)
- [mordomo-whisper-asr](https://github.com/AslamSys/mordomo-whisper-asr)
- [mordomo-speaker-id-diarization](https://github.com/AslamSys/mordomo-speaker-id-diarization)
- [mordomo-source-separation](https://github.com/AslamSys/mordomo-source-separation)
- [mordomo-core-gateway](https://github.com/AslamSys/mordomo-core-gateway)
- [mordomo-orchestrator](https://github.com/AslamSys/mordomo-orchestrator)
- [mordomo-tts-engine](https://github.com/AslamSys/mordomo-tts-engine)
- [mordomo-system-watchdog](https://github.com/AslamSys/mordomo-system-watchdog)
- [mordomo-dashboard-ui](https://github.com/AslamSys/mordomo-dashboard-ui)
- [mordomo-openclaw-agent](https://github.com/AslamSys/mordomo-openclaw-agent)
- [mordomo-action-dispatcher](https://github.com/AslamSys/mordomo-action-dispatcher)
- [mordomo-skills-runner](https://github.com/AslamSys/mordomo-skills-runner)

---
# Instalar separadamente: curl https://ollama.ai/install.sh | sh

## 🔗 Navegação

**[🏠 AslamSys](https://github.com/AslamSys)** → **[📚 _system](https://github.com/AslamSys/_system)** → **[📂 Aslam (Orange Pi 5 16GB)](https://github.com/AslamSys/_system/blob/main/hardware/mordomo%20-%20(orange-pi-5-16gb)/README.md)** → **mordomo-brain**

### Containers Relacionados (aslam)
- [mordomo-audio-bridge](https://github.com/AslamSys/mordomo-audio-bridge)
- [mordomo-audio-capture-vad](https://github.com/AslamSys/mordomo-audio-capture-vad)
- [mordomo-wake-word-detector](https://github.com/AslamSys/mordomo-wake-word-detector)
- [mordomo-speaker-verification](https://github.com/AslamSys/mordomo-speaker-verification)
- [mordomo-whisper-asr](https://github.com/AslamSys/mordomo-whisper-asr)
- [mordomo-speaker-id-diarization](https://github.com/AslamSys/mordomo-speaker-id-diarization)
- [mordomo-source-separation](https://github.com/AslamSys/mordomo-source-separation)
- [mordomo-core-gateway](https://github.com/AslamSys/mordomo-core-gateway)
- [mordomo-orchestrator](https://github.com/AslamSys/mordomo-orchestrator)
- [mordomo-tts-engine](https://github.com/AslamSys/mordomo-tts-engine)
- [mordomo-system-watchdog](https://github.com/AslamSys/mordomo-system-watchdog)
- [mordomo-dashboard-ui](https://github.com/AslamSys/mordomo-dashboard-ui)
- [mordomo-openclaw-agent](https://github.com/AslamSys/mordomo-openclaw-agent)
- [mordomo-action-dispatcher](https://github.com/AslamSys/mordomo-action-dispatcher)
- [mordomo-skills-runner](https://github.com/AslamSys/mordomo-skills-runner)

---
# Baixar modelo: ollama pull qwen2.5:3b

## 🔗 Navegação

**[🏠 AslamSys](https://github.com/AslamSys)** → **[📚 _system](https://github.com/AslamSys/_system)** → **[📂 Aslam (Orange Pi 5 16GB)](https://github.com/AslamSys/_system/blob/main/hardware/mordomo%20-%20(orange-pi-5-16gb)/README.md)** → **mordomo-brain**

### Containers Relacionados (aslam)
- [mordomo-audio-bridge](https://github.com/AslamSys/mordomo-audio-bridge)
- [mordomo-audio-capture-vad](https://github.com/AslamSys/mordomo-audio-capture-vad)
- [mordomo-wake-word-detector](https://github.com/AslamSys/mordomo-wake-word-detector)
- [mordomo-speaker-verification](https://github.com/AslamSys/mordomo-speaker-verification)
- [mordomo-whisper-asr](https://github.com/AslamSys/mordomo-whisper-asr)
- [mordomo-speaker-id-diarization](https://github.com/AslamSys/mordomo-speaker-id-diarization)
- [mordomo-source-separation](https://github.com/AslamSys/mordomo-source-separation)
- [mordomo-core-gateway](https://github.com/AslamSys/mordomo-core-gateway)
- [mordomo-orchestrator](https://github.com/AslamSys/mordomo-orchestrator)
- [mordomo-tts-engine](https://github.com/AslamSys/mordomo-tts-engine)
- [mordomo-system-watchdog](https://github.com/AslamSys/mordomo-system-watchdog)
- [mordomo-dashboard-ui](https://github.com/AslamSys/mordomo-dashboard-ui)
- [mordomo-openclaw-agent](https://github.com/AslamSys/mordomo-openclaw-agent)
- [mordomo-action-dispatcher](https://github.com/AslamSys/mordomo-action-dispatcher)
- [mordomo-skills-runner](https://github.com/AslamSys/mordomo-skills-runner)

---
```

---

## 🔧 Troubleshooting

### Ollama não conecta
```bash
# Verificar se Ollama está rodando

## 🔗 Navegação

**[🏠 AslamSys](https://github.com/AslamSys)** → **[📚 _system](https://github.com/AslamSys/_system)** → **[📂 Aslam (Orange Pi 5 16GB)](https://github.com/AslamSys/_system/blob/main/hardware/mordomo%20-%20(orange-pi-5-16gb)/README.md)** → **mordomo-brain**

### Containers Relacionados (aslam)
- [mordomo-audio-bridge](https://github.com/AslamSys/mordomo-audio-bridge)
- [mordomo-audio-capture-vad](https://github.com/AslamSys/mordomo-audio-capture-vad)
- [mordomo-wake-word-detector](https://github.com/AslamSys/mordomo-wake-word-detector)
- [mordomo-speaker-verification](https://github.com/AslamSys/mordomo-speaker-verification)
- [mordomo-whisper-asr](https://github.com/AslamSys/mordomo-whisper-asr)
- [mordomo-speaker-id-diarization](https://github.com/AslamSys/mordomo-speaker-id-diarization)
- [mordomo-source-separation](https://github.com/AslamSys/mordomo-source-separation)
- [mordomo-core-gateway](https://github.com/AslamSys/mordomo-core-gateway)
- [mordomo-orchestrator](https://github.com/AslamSys/mordomo-orchestrator)
- [mordomo-tts-engine](https://github.com/AslamSys/mordomo-tts-engine)
- [mordomo-system-watchdog](https://github.com/AslamSys/mordomo-system-watchdog)
- [mordomo-dashboard-ui](https://github.com/AslamSys/mordomo-dashboard-ui)
- [mordomo-openclaw-agent](https://github.com/AslamSys/mordomo-openclaw-agent)
- [mordomo-action-dispatcher](https://github.com/AslamSys/mordomo-action-dispatcher)
- [mordomo-skills-runner](https://github.com/AslamSys/mordomo-skills-runner)

---
curl http://localhost:11434/api/tags

# Baixar modelo

## 🔗 Navegação

**[🏠 AslamSys](https://github.com/AslamSys)** → **[📚 _system](https://github.com/AslamSys/_system)** → **[📂 Aslam (Orange Pi 5 16GB)](https://github.com/AslamSys/_system/blob/main/hardware/mordomo%20-%20(orange-pi-5-16gb)/README.md)** → **mordomo-brain**

### Containers Relacionados (aslam)
- [mordomo-audio-bridge](https://github.com/AslamSys/mordomo-audio-bridge)
- [mordomo-audio-capture-vad](https://github.com/AslamSys/mordomo-audio-capture-vad)
- [mordomo-wake-word-detector](https://github.com/AslamSys/mordomo-wake-word-detector)
- [mordomo-speaker-verification](https://github.com/AslamSys/mordomo-speaker-verification)
- [mordomo-whisper-asr](https://github.com/AslamSys/mordomo-whisper-asr)
- [mordomo-speaker-id-diarization](https://github.com/AslamSys/mordomo-speaker-id-diarization)
- [mordomo-source-separation](https://github.com/AslamSys/mordomo-source-separation)
- [mordomo-core-gateway](https://github.com/AslamSys/mordomo-core-gateway)
- [mordomo-orchestrator](https://github.com/AslamSys/mordomo-orchestrator)
- [mordomo-tts-engine](https://github.com/AslamSys/mordomo-tts-engine)
- [mordomo-system-watchdog](https://github.com/AslamSys/mordomo-system-watchdog)
- [mordomo-dashboard-ui](https://github.com/AslamSys/mordomo-dashboard-ui)
- [mordomo-openclaw-agent](https://github.com/AslamSys/mordomo-openclaw-agent)
- [mordomo-action-dispatcher](https://github.com/AslamSys/mordomo-action-dispatcher)
- [mordomo-skills-runner](https://github.com/AslamSys/mordomo-skills-runner)

---
ollama pull qwen2.5:3b

# Verificar logs

## 🔗 Navegação

**[🏠 AslamSys](https://github.com/AslamSys)** → **[📚 _system](https://github.com/AslamSys/_system)** → **[📂 Aslam (Orange Pi 5 16GB)](https://github.com/AslamSys/_system/blob/main/hardware/mordomo%20-%20(orange-pi-5-16gb)/README.md)** → **mordomo-brain**

### Containers Relacionados (aslam)
- [mordomo-audio-bridge](https://github.com/AslamSys/mordomo-audio-bridge)
- [mordomo-audio-capture-vad](https://github.com/AslamSys/mordomo-audio-capture-vad)
- [mordomo-wake-word-detector](https://github.com/AslamSys/mordomo-wake-word-detector)
- [mordomo-speaker-verification](https://github.com/AslamSys/mordomo-speaker-verification)
- [mordomo-whisper-asr](https://github.com/AslamSys/mordomo-whisper-asr)
- [mordomo-speaker-id-diarization](https://github.com/AslamSys/mordomo-speaker-id-diarization)
- [mordomo-source-separation](https://github.com/AslamSys/mordomo-source-separation)
- [mordomo-core-gateway](https://github.com/AslamSys/mordomo-core-gateway)
- [mordomo-orchestrator](https://github.com/AslamSys/mordomo-orchestrator)
- [mordomo-tts-engine](https://github.com/AslamSys/mordomo-tts-engine)
- [mordomo-system-watchdog](https://github.com/AslamSys/mordomo-system-watchdog)
- [mordomo-dashboard-ui](https://github.com/AslamSys/mordomo-dashboard-ui)
- [mordomo-openclaw-agent](https://github.com/AslamSys/mordomo-openclaw-agent)
- [mordomo-action-dispatcher](https://github.com/AslamSys/mordomo-action-dispatcher)
- [mordomo-skills-runner](https://github.com/AslamSys/mordomo-skills-runner)

---
ollama logs
```

### Respostas muito lentas
```yaml
# Usar modelo menor

## 🔗 Navegação

**[🏠 AslamSys](https://github.com/AslamSys)** → **[📚 _system](https://github.com/AslamSys/_system)** → **[📂 Aslam (Orange Pi 5 16GB)](https://github.com/AslamSys/_system/blob/main/hardware/mordomo%20-%20(orange-pi-5-16gb)/README.md)** → **mordomo-brain**

### Containers Relacionados (aslam)
- [mordomo-audio-bridge](https://github.com/AslamSys/mordomo-audio-bridge)
- [mordomo-audio-capture-vad](https://github.com/AslamSys/mordomo-audio-capture-vad)
- [mordomo-wake-word-detector](https://github.com/AslamSys/mordomo-wake-word-detector)
- [mordomo-speaker-verification](https://github.com/AslamSys/mordomo-speaker-verification)
- [mordomo-whisper-asr](https://github.com/AslamSys/mordomo-whisper-asr)
- [mordomo-speaker-id-diarization](https://github.com/AslamSys/mordomo-speaker-id-diarization)
- [mordomo-source-separation](https://github.com/AslamSys/mordomo-source-separation)
- [mordomo-core-gateway](https://github.com/AslamSys/mordomo-core-gateway)
- [mordomo-orchestrator](https://github.com/AslamSys/mordomo-orchestrator)
- [mordomo-tts-engine](https://github.com/AslamSys/mordomo-tts-engine)
- [mordomo-system-watchdog](https://github.com/AslamSys/mordomo-system-watchdog)
- [mordomo-dashboard-ui](https://github.com/AslamSys/mordomo-dashboard-ui)
- [mordomo-openclaw-agent](https://github.com/AslamSys/mordomo-openclaw-agent)
- [mordomo-action-dispatcher](https://github.com/AslamSys/mordomo-action-dispatcher)
- [mordomo-skills-runner](https://github.com/AslamSys/mordomo-skills-runner)

---
ollama.model: "qwen2.5:3b"  # ao invés de 7b

# Reduzir tokens

## 🔗 Navegação

**[🏠 AslamSys](https://github.com/AslamSys)** → **[📚 _system](https://github.com/AslamSys/_system)** → **[📂 Aslam (Orange Pi 5 16GB)](https://github.com/AslamSys/_system/blob/main/hardware/mordomo%20-%20(orange-pi-5-16gb)/README.md)** → **mordomo-brain**

### Containers Relacionados (aslam)
- [mordomo-audio-bridge](https://github.com/AslamSys/mordomo-audio-bridge)
- [mordomo-audio-capture-vad](https://github.com/AslamSys/mordomo-audio-capture-vad)
- [mordomo-wake-word-detector](https://github.com/AslamSys/mordomo-wake-word-detector)
- [mordomo-speaker-verification](https://github.com/AslamSys/mordomo-speaker-verification)
- [mordomo-whisper-asr](https://github.com/AslamSys/mordomo-whisper-asr)
- [mordomo-speaker-id-diarization](https://github.com/AslamSys/mordomo-speaker-id-diarization)
- [mordomo-source-separation](https://github.com/AslamSys/mordomo-source-separation)
- [mordomo-core-gateway](https://github.com/AslamSys/mordomo-core-gateway)
- [mordomo-orchestrator](https://github.com/AslamSys/mordomo-orchestrator)
- [mordomo-tts-engine](https://github.com/AslamSys/mordomo-tts-engine)
- [mordomo-system-watchdog](https://github.com/AslamSys/mordomo-system-watchdog)
- [mordomo-dashboard-ui](https://github.com/AslamSys/mordomo-dashboard-ui)
- [mordomo-openclaw-agent](https://github.com/AslamSys/mordomo-openclaw-agent)
- [mordomo-action-dispatcher](https://github.com/AslamSys/mordomo-action-dispatcher)
- [mordomo-skills-runner](https://github.com/AslamSys/mordomo-skills-runner)

---
ollama.num_predict: 256

# Aumentar fallback para cloud

## 🔗 Navegação

**[🏠 AslamSys](https://github.com/AslamSys)** → **[📚 _system](https://github.com/AslamSys/_system)** → **[📂 Aslam (Orange Pi 5 16GB)](https://github.com/AslamSys/_system/blob/main/hardware/mordomo%20-%20(orange-pi-5-16gb)/README.md)** → **mordomo-brain**

### Containers Relacionados (aslam)
- [mordomo-audio-bridge](https://github.com/AslamSys/mordomo-audio-bridge)
- [mordomo-audio-capture-vad](https://github.com/AslamSys/mordomo-audio-capture-vad)
- [mordomo-wake-word-detector](https://github.com/AslamSys/mordomo-wake-word-detector)
- [mordomo-speaker-verification](https://github.com/AslamSys/mordomo-speaker-verification)
- [mordomo-whisper-asr](https://github.com/AslamSys/mordomo-whisper-asr)
- [mordomo-speaker-id-diarization](https://github.com/AslamSys/mordomo-speaker-id-diarization)
- [mordomo-source-separation](https://github.com/AslamSys/mordomo-source-separation)
- [mordomo-core-gateway](https://github.com/AslamSys/mordomo-core-gateway)
- [mordomo-orchestrator](https://github.com/AslamSys/mordomo-orchestrator)
- [mordomo-tts-engine](https://github.com/AslamSys/mordomo-tts-engine)
- [mordomo-system-watchdog](https://github.com/AslamSys/mordomo-system-watchdog)
- [mordomo-dashboard-ui](https://github.com/AslamSys/mordomo-dashboard-ui)
- [mordomo-openclaw-agent](https://github.com/AslamSys/mordomo-openclaw-agent)
- [mordomo-action-dispatcher](https://github.com/AslamSys/mordomo-action-dispatcher)
- [mordomo-skills-runner](https://github.com/AslamSys/mordomo-skills-runner)

---
strategy.local_timeout: 3000  # 3s
```

### Cloud API erros
```yaml
# Verificar keys

## 🔗 Navegação

**[🏠 AslamSys](https://github.com/AslamSys)** → **[📚 _system](https://github.com/AslamSys/_system)** → **[📂 Aslam (Orange Pi 5 16GB)](https://github.com/AslamSys/_system/blob/main/hardware/mordomo%20-%20(orange-pi-5-16gb)/README.md)** → **mordomo-brain**

### Containers Relacionados (aslam)
- [mordomo-audio-bridge](https://github.com/AslamSys/mordomo-audio-bridge)
- [mordomo-audio-capture-vad](https://github.com/AslamSys/mordomo-audio-capture-vad)
- [mordomo-wake-word-detector](https://github.com/AslamSys/mordomo-wake-word-detector)
- [mordomo-speaker-verification](https://github.com/AslamSys/mordomo-speaker-verification)
- [mordomo-whisper-asr](https://github.com/AslamSys/mordomo-whisper-asr)
- [mordomo-speaker-id-diarization](https://github.com/AslamSys/mordomo-speaker-id-diarization)
- [mordomo-source-separation](https://github.com/AslamSys/mordomo-source-separation)
- [mordomo-core-gateway](https://github.com/AslamSys/mordomo-core-gateway)
- [mordomo-orchestrator](https://github.com/AslamSys/mordomo-orchestrator)
- [mordomo-tts-engine](https://github.com/AslamSys/mordomo-tts-engine)
- [mordomo-system-watchdog](https://github.com/AslamSys/mordomo-system-watchdog)
- [mordomo-dashboard-ui](https://github.com/AslamSys/mordomo-dashboard-ui)
- [mordomo-openclaw-agent](https://github.com/AslamSys/mordomo-openclaw-agent)
- [mordomo-action-dispatcher](https://github.com/AslamSys/mordomo-action-dispatcher)
- [mordomo-skills-runner](https://github.com/AslamSys/mordomo-skills-runner)

---
echo $OPENAI_API_KEY

# Verificar rate limits

## 🔗 Navegação

**[🏠 AslamSys](https://github.com/AslamSys)** → **[📚 _system](https://github.com/AslamSys/_system)** → **[📂 Aslam (Orange Pi 5 16GB)](https://github.com/AslamSys/_system/blob/main/hardware/mordomo%20-%20(orange-pi-5-16gb)/README.md)** → **mordomo-brain**

### Containers Relacionados (aslam)
- [mordomo-audio-bridge](https://github.com/AslamSys/mordomo-audio-bridge)
- [mordomo-audio-capture-vad](https://github.com/AslamSys/mordomo-audio-capture-vad)
- [mordomo-wake-word-detector](https://github.com/AslamSys/mordomo-wake-word-detector)
- [mordomo-speaker-verification](https://github.com/AslamSys/mordomo-speaker-verification)
- [mordomo-whisper-asr](https://github.com/AslamSys/mordomo-whisper-asr)
- [mordomo-speaker-id-diarization](https://github.com/AslamSys/mordomo-speaker-id-diarization)
- [mordomo-source-separation](https://github.com/AslamSys/mordomo-source-separation)
- [mordomo-core-gateway](https://github.com/AslamSys/mordomo-core-gateway)
- [mordomo-orchestrator](https://github.com/AslamSys/mordomo-orchestrator)
- [mordomo-tts-engine](https://github.com/AslamSys/mordomo-tts-engine)
- [mordomo-system-watchdog](https://github.com/AslamSys/mordomo-system-watchdog)
- [mordomo-dashboard-ui](https://github.com/AslamSys/mordomo-dashboard-ui)
- [mordomo-openclaw-agent](https://github.com/AslamSys/mordomo-openclaw-agent)
- [mordomo-action-dispatcher](https://github.com/AslamSys/mordomo-action-dispatcher)
- [mordomo-skills-runner](https://github.com/AslamSys/mordomo-skills-runner)

---
# Adicionar retry com backoff

## 🔗 Navegação

**[🏠 AslamSys](https://github.com/AslamSys)** → **[📚 _system](https://github.com/AslamSys/_system)** → **[📂 Aslam (Orange Pi 5 16GB)](https://github.com/AslamSys/_system/blob/main/hardware/mordomo%20-%20(orange-pi-5-16gb)/README.md)** → **mordomo-brain**

### Containers Relacionados (aslam)
- [mordomo-audio-bridge](https://github.com/AslamSys/mordomo-audio-bridge)
- [mordomo-audio-capture-vad](https://github.com/AslamSys/mordomo-audio-capture-vad)
- [mordomo-wake-word-detector](https://github.com/AslamSys/mordomo-wake-word-detector)
- [mordomo-speaker-verification](https://github.com/AslamSys/mordomo-speaker-verification)
- [mordomo-whisper-asr](https://github.com/AslamSys/mordomo-whisper-asr)
- [mordomo-speaker-id-diarization](https://github.com/AslamSys/mordomo-speaker-id-diarization)
- [mordomo-source-separation](https://github.com/AslamSys/mordomo-source-separation)
- [mordomo-core-gateway](https://github.com/AslamSys/mordomo-core-gateway)
- [mordomo-orchestrator](https://github.com/AslamSys/mordomo-orchestrator)
- [mordomo-tts-engine](https://github.com/AslamSys/mordomo-tts-engine)
- [mordomo-system-watchdog](https://github.com/AslamSys/mordomo-system-watchdog)
- [mordomo-dashboard-ui](https://github.com/AslamSys/mordomo-dashboard-ui)
- [mordomo-openclaw-agent](https://github.com/AslamSys/mordomo-openclaw-agent)
- [mordomo-action-dispatcher](https://github.com/AslamSys/mordomo-action-dispatcher)
- [mordomo-skills-runner](https://github.com/AslamSys/mordomo-skills-runner)

---
```

---

## 🔐 Vault Integration

O `mordomo-brain` usa APIs de LLM cloud (OpenAI, Anthropic, Groq, Google). Essas API keys **não ficam em env vars** — são obtidas do `mordomo-vault` via `service` auth na inicialização do container.

```yaml
Credenciais gerenciadas pelo vault:
  - openai_api_key
  - anthropic_api_key
  - groq_api_key
  - google_ai_api_key

Auth mode: service
Módulo token: ${VAULT_MODULE_TOKEN}  # emitido pelo vault CLI no bootstrap
```

As keys são carregadas uma vez na inicialização e mantidas em memória. Em caso de rotação, o vault publica `mordomo.vault.secret.rotated` e o brain recarrega.

Veja: [mordomo-vault](https://github.com/AslamSys/mordomo-vault)

---

**Versão:** 1.0  
**Última atualização:** 14/04/2026

---

## 🗂️ Estrutura do Repositório

```
mordomo-brain/
├── src/
│   ├── config.py      # Env vars: NATS, LiteLLM, Redis, Qdrant, models
│   ├── context.py     # Redis db1 — histórico de conversa por speaker_id (TTL 30min)
│   ├── llm.py         # HTTP client para LiteLLM gateway — roteamento simple/complex/stakes
│   ├── rag.py         # Qdrant search + upsert (opcional, degrada graciosamente)
│   ├── actions.py     # Parser de [ACTION: {...}] no texto do LLM
│   ├── handlers.py    # NATS handler: mordomo.brain.generate (request/reply)
│   └── main.py        # Entrypoint asyncio + reconnect loop
├── requirements.txt    # nats-py, redis, httpx
├── Dockerfile
├── docker-compose.yml
└── .env.example
```

## 🔌 Infraestrutura Utilizada

| Serviço infra | Uso | DB / Endpoint |
|---|---|---|
| **NATS** | Subscribe `mordomo.brain.generate`, publish `mordomo.brain.action.*` | — |
| **Redis** | Histórico de conversa por speaker_id | db1 (mordomo-general) |
| **Qdrant** | RAG semântico — busca e armazena trechos de conversa | collection `mordomo_conversations` |
| **LiteLLM Gateway** | Proxy para OpenAI / Anthropic / Groq (API keys ficam no gateway) | porta 4000 |

## 🔄 Roteamento de Modelos

| Tipo de query | Modelo virtual | Exemplo |
|---|---|---|
| Simples (IoT, hora, etc) | `mordomo-simple` → gemini-2.0-flash | "Acende a luz da sala" |
| Complexa (análise, raciocínio) | `mordomo-complex` → gpt-4o-mini | "Explique meu extrato do mês" |
| Alto risco (finanças, senhas) | `mordomo-stakes` → claude-3.5-haiku | "Faz um PIX de R$500" |

Fallback automático para `mordomo-simple` se modelo primário falhar.
