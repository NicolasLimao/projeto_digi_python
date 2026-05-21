# Digi RAG - Sistema de Suporte Inteligente para Digisac

> **RAG (Retrieval-Augmented Generation) 100% em Python** com integração Discord e Supabase

## 🚀 Quick Start

### 1. Instalar Dependências

```bash
# Windows (no prompt do projeto)
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configurar Credenciais

As credenciais já estão em `.env` com valores reais dos workflows n8n.

Caso queira usar outras, edite o arquivo `.env`:
```env
OPENAI_API_KEY=sk-proj-xxxxx
SUPABASE_URL=https://xxxxx.supabase.co
SUPABASE_ANON_KEY=xxxxx
DISCORD_BOT_TOKEN=xxxxx
```

### 3. Rodar Servidor

**Opção 1: PowerShell (Recomendado)**
```powershell
# Rodar do VS Code ou PowerShell
.\run.ps1
```

**Opção 2: Batch (Windows)**
```cmd
run.bat
```

**Opção 3: Direto**
```bash
# Com venv ativado
python main.py
```

### 4. Testar

```bash
# Health check
curl http://localhost:8000/health

# Swagger UI
http://localhost:8000/docs

# RAG Query
curl -X POST http://localhost:8000/api/rag/query \
  -H "Content-Type: application/json" \
  -d '{"query":"Como fazer backup no Digisac?"}' \
  ?user_id=user_123
```

---

## 📋 Arquitetura

### Pipeline de Processamento

```
Discord DM
    ↓
FastAPI Endpoint (/api/rag/query)
    ↓
ClassifierAgent (GPT-4o-mini)
    ↓
ScopeValidatorAgent (GPT-4o-mini)
    ↓
RAGAgent {
  - Embeddings (text-embedding-3-small)
  - Vector Search (Supabase pgvector)
  - LLM Generation (GPT-4o-mini)
}
    ↓
FormatterAgent (Mode: orientacao/resposta-cliente/bug)
    ↓
HistoryService (Persistência Supabase)
    ↓
Discord Response
```

### Componentes

| Componente | Arquivo | Descrição |
|---|---|---|
| **OpenAIService** | `src/services/openai_service.py` | Integração com OpenAI APIs |
| **SupabaseService** | `src/services/supabase_service.py` | Vector search + persistência |
| **HistoryService** | `src/services/history_service.py` | Histórico de conversas |
| **RAGAgent** | `src/agents/rag_agent.py` | Orquestração RAG |
| **FormatterAgent** | `src/agents/formatter_agent.py` | Formatação de resposta |
| **RAGPipeline** | `src/pipeline/rag_pipeline.py` | Pipeline completo |
| **API Routes** | `src/api/rag_routes.py` | Endpoints FastAPI |

---

## 🔌 Endpoints API

### RAG Query
```
POST /api/rag/query?user_id=<user_id>
Content-Type: application/json

Body:
{
  "query": "Como fazer backup?",
  "mode": "orientacao"  # Opcional: orientacao|resposta-cliente|bug
}

Response:
{
  "response": "resposta formatada",
  "mode": "orientacao",
  "score": 0.85,
  "chunks_used": 2,
  "processing_time_ms": 1500
}
```

### History APIs
```
# Obter histórico do usuário
GET /api/history/user/{user_id}?limit=10

# Obter contexto formatado para prompt
POST /api/history/context
{
  "user_id": "user_123",
  "limit": 5
}

# Salvar interação
POST /api/history/save?user_id=user_123&query=...&resposta=...

# Limpar histórico antigo
DELETE /api/history/cleanup?days_to_keep=30
```

### Health Check
```
GET /health
Response: {"status": "ok", "service": "digi-rag"}
```

---

## 🧪 Testes

```bash
# Todos os testes
pytest tests/ -v

# Um arquivo específico
pytest tests/agents/test_rag_agent.py -v

# Com cobertura
pytest tests/ --cov=src

# Status: 79/79 ✅
```

---

## 🔧 Configuração

### Variáveis de Ambiente (.env)

```env
# OpenAI
OPENAI_API_KEY=sk-proj-xxxxx

# Supabase
SUPABASE_URL=https://xxxxx.supabase.co
SUPABASE_ANON_KEY=xxxxx

# Discord
DISCORD_BOT_TOKEN=xxxxx
DISCORD_GUILD_ID=1491637126440288377
DISCORD_CHANNEL_DUVIDAS=1491637352513142914
DISCORD_CHANNEL_LOGS=1491637401024729231
DISCORD_CHANNEL_GAPS=1491637453298208902

# RAG API
RAG_API_URL=http://localhost:8000/api/rag/query

# Configurações
LOG_LEVEL=INFO
ENVIRONMENT=production
SCORE_THRESHOLD=0.65
MAX_CHUNKS=10
HISTORY_ENABLED=true
HISTORY_RETENTION_DAYS=90
DISCORD_DM_ENABLED=true
```

---

## 📊 Performance

| Operação | Tempo |
|---|---|
| Classificação | 200-500ms |
| Validação de escopo | 200-500ms |
| Embeddings | 100-300ms |
| Vector search | 50-200ms |
| LLM generation | 1-3s |
| **Total e2e** | **2-5s** |

---

## 🐛 Troubleshooting

### ModuleNotFoundError: No module named 'fastapi'
```bash
# Ativar venv e reinstalar
.venv\Scripts\activate
pip install -r requirements.txt
```

### "No Supabase client" / "No API key"
O sistema usa **mock fallback** quando sem credenciais. Adicione credenciais reais em `.env`.

### Conexão recusada em http://localhost:8000
O servidor pode estar desligado. Execute `python main.py` na pasta do projeto.

### Discord DM não funciona
1. Verificar `DISCORD_DM_ENABLED=true` em `.env`
2. Verificar `RAG_API_URL` apontando para o servidor
3. Verificar logs do Discord bot

---

## 📚 Documentação Detalhada

- [RAG Implementation](docs/RAG_IMPLEMENTATION.md) - Arquitetura e detalhes técnicos
- [History Feature](docs/HISTORY_FEATURE.md) - Sistema de histórico
- [n8n Integration](contexto/WORKFLOW_HISTORY_INTEGRATION.md) - Integração com n8n

---

## 🚢 Deployment

### Docker (Recomendado)
```bash
docker build -t digi-rag .
docker run -p 8000:8000 --env-file .env digi-rag
```

### Direct (Python)
```bash
pip install -r requirements.txt
python main.py
```

### Produção (Uvicorn)
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

---

## 📝 Commits Principais

| Hash | Mensagem |
|---|---|
| `4f06dd9` | feat: implement complete RAG system 100% in Python |
| `edf6cac` | config: add real API credentials from n8n workflows |
| `72e6153` | fix: update Settings config to support new RAG fields |

---

## 🔗 Links Úteis

- **Swagger API Docs**: http://localhost:8000/docs
- **OpenAI API**: https://platform.openai.com/docs
- **Supabase Vector Search**: https://supabase.com/docs/guides/database/vector
- **FastAPI**: https://fastapi.tiangolo.com/
- **Discord.py**: https://discordpy.readthedocs.io/

---

## 📄 License

Proprietary - Digisac 2026

---

**Status**: ✅ Production-Ready com Credenciais Reais
**Versão**: 1.0.0
**Última atualização**: 2026-05-21
