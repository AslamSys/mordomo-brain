# 🧠 Mordomo Brain (LLM)

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

### LLM Local
**Ollama** - Runtime para modelos locais
- Suporta Qwen, Llama, Mistral, Phi
- API compatível com OpenAI
- Otimizado para CPU/GPU
- Quantização automática

**Modelos Recomendados:**
```yaml
Primary: qwen2.5:3b (3GB RAM, rápido, português ok)
Fallback: llama3.2:3b (backup)
Heavy: qwen2.5:7b (melhor qualidade, mais lento)
```

### LLM Cloud (Fallback)
```yaml
OpenAI: gpt-4o-mini, gpt-4o
Anthropic: claude-3-haiku, claude-3.5-sonnet
Google: gemini-1.5-flash, gemini-1.5-pro
Groq: llama-3.1-70b (rápido)
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
Performance Local (Qwen 2.5 3B):
  CPU: 40-60% (4 cores ARM)
  RAM: ~ 3.5 GB
  Tokens/s: 15-25 (generation)
  Latency: 1-3s (resposta completa)
  Context Window: 32k tokens

Performance Cloud (GPT-4o-mini):
  API Call Latency: 500-1500ms
  Cost: $0.15/1M input tokens
  Context Window: 128k tokens

Strategies:
  local-first: 90% uso local, 10% cloud
  cloud-only: 100% cloud (tarefas complexas)
  mixed: Local reasoning + Cloud refinement
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
    model: "llama-3.1-70b-versatile"

strategy:
  default: "local-first"
  
  # Condições para cloud
  use_cloud_if:
    - query_complexity > 0.8
    - local_confidence < 0.6
    - requires_real_time_data: true
    - user_request_contains: ["pesquise", "busque na internet"]
  
  # Fallback automático
  auto_fallback:
    enabled: true
    local_timeout: 5000  # ms
    local_error_threshold: 3

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
llm_requests_total{model, strategy}
llm_tokens_total{model, type}  # type: input/output
llm_latency_seconds{model, percentile}
llm_cost_usd{model}

# Performance
llm_cache_hits_total
llm_cache_misses_total
llm_fallback_total{from, to}

# Qualidade
llm_confidence_avg{model}
llm_intent_accuracy

# Ações
actions_executed_total{type}
actions_failed_total{type, reason}
```

---

## 🐳 Docker

```dockerfile
FROM python:3.11-slim

# Ollama (será instalado no host, não no container)
# Este container apenas se conecta ao Ollama via HTTP

WORKDIR /app

# Dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download embedding model
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')"

# Application
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
# Instalar separadamente: curl https://ollama.ai/install.sh | sh
# Baixar modelo: ollama pull qwen2.5:3b
```

---

## 🔧 Troubleshooting

### Ollama não conecta
```bash
# Verificar se Ollama está rodando
curl http://localhost:11434/api/tags

# Baixar modelo
ollama pull qwen2.5:3b

# Verificar logs
ollama logs
```

### Respostas muito lentas
```yaml
# Usar modelo menor
ollama.model: "qwen2.5:3b"  # ao invés de 7b

# Reduzir tokens
ollama.num_predict: 256

# Aumentar fallback para cloud
strategy.local_timeout: 3000  # 3s
```

### Cloud API erros
```yaml
# Verificar keys
echo $OPENAI_API_KEY

# Verificar rate limits
# Adicionar retry com backoff
```

---

**Versão:** 1.0  
**Última atualização:** 27/11/2025
