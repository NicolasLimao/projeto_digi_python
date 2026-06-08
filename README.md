# Digi — Agente RAG para Suporte Técnico Interno

Assistente conversacional baseado em IA generativa que responde dúvidas técnicas via Discord, consultando a documentação completa de uma plataforma de atendimento multicanal (Digisac). Reduz o tempo gasto buscando informações em manuais a uma conversa de poucos segundos, com respostas fundamentadas em fontes oficiais.

Construído end-to-end: arquitetura RAG (Retrieval-Augmented Generation) em Python, bot Discord em Node.js, banco vetorial em PostgreSQL com pgvector via Supabase, modelos da OpenAI para embeddings e geração, ingestão de PDF com PyMuPDF e fallback para Mistral OCR, hospedado em produção na SquareCloud com auto-deploy via GitHub.

---

## O problema

Analistas de suporte N1 perdem tempo significativo buscando informações na documentação ou escalando dúvidas que poderiam ser resolvidas com acesso rápido ao manual. Cada escalação desnecessária atrasa o atendimento ao cliente final e custa tempo do time de N2.

## A solução

Um agente acessível pelo próprio canal de trabalho da equipe (Discord, em canal aberto ou em DM privada), com três comportamentos detectados automaticamente a partir do conteúdo da mensagem:

- **Orientação** — tom de colega técnico, com passos ou explicação conceitual
- **Resposta para o cliente** — texto pronto, profissional, sem expor processos internos
- **Bug** — checklist estruturado de investigação

A intenção é classificada pelo próprio modelo a partir do conteúdo da mensagem.

---

## Arquitetura

```
Discord
   |
   | mensagem (canal ou DM)
   v
Bot Node.js
   |
   | HTTP POST /api/rag/query
   v
API Python (FastAPI)
   |
   |-> Classifier          (em paralelo via asyncio.gather)
   |-> Scope Validator
   |-> Retrieval
   |       |
   |       |-> rewrite query (condicional)
   |       |-> embedding (OpenAI)
   |       |-> hybrid search (Supabase pgvector)
   |       |-> rerank (LLM)
   |
   v
Generation Prompt -> OpenAI gpt-4o-mini -> Resposta
   |
   v
Bot publica no Discord + reacoes de feedback (positivo/negativo)
   |
   v
Historico salvo no Supabase (analytics + identificacao de gaps)
```

As três tarefas iniciais (classify, validate, retrieve) executam concorrentemente via `asyncio.gather`, escondendo a latência da classificação atrás da busca. A geração só roda quando o pool reranqueado está pronto.

---

## Stack

| Camada | Tecnologia |
|--------|-----------|
| API | Python, FastAPI, Uvicorn |
| Validação | Pydantic v2 |
| Bot | Node.js, discord.js v14 |
| Banco | Supabase (PostgreSQL + pgvector) |
| Embeddings | OpenAI `text-embedding-3-small` (1536 dimensões) |
| Geração | OpenAI `gpt-4o-mini` |
| Ingestão PDF | PyMuPDF (texto selecionável) + Mistral OCR (fallback para PDF imagem) |
| Hospedagem | SquareCloud, com auto-deploy via GitHub |

---

## Técnicas de RAG implementadas

**Busca híbrida ponderada.** Combinação 50/50 de busca semântica (embeddings) e full-text (palavra-chave) via função PL/pgSQL no Supabase. Pega o melhor dos dois mundos: relevância por significado e precisão de termos técnicos.

**Reescrita de query antes do embedding.** Uma mensagem ruidosa como `Dúvida do cliente — Obs.1 — Obs.2 — Tintim...` é destilada para a intenção real (`disparos de webhook com conteúdo de mensagem`) antes de virar vetor. Em follow-ups, usa o histórico da conversa para resolver referências (`e como faço isso?` se transforma em uma query autossuficiente).

**Reranking com pool ampliado.** Recupera 15 candidatos via busca híbrida, reordena por relevância real usando o LLM, mantém os 10 melhores para a geração. Aumenta recall sem inflar o contexto.

**Gate adaptativo de reescrita.** Perguntas curtas e diretas pulam a reescrita (evita chamada extra de modelo); perguntas longas, com histórico ou com notas internas passam pelo pipeline completo.

**Memória multi-turno por usuário.** Últimas 4 trocas do mesmo usuário (janela de 60 minutos) são injetadas no prompt, permitindo conversas reais em DM. Cada usuário tem seu próprio contexto isolado.

**Loop de feedback como dataset rotulado.** Cada resposta do bot recebe reações de aprovação ou reprovação. O voto vai para a tabela `historico_digi` junto com a pergunta original, a query reescrita, as fontes retornadas, o modo, o score de recuperação e o canal. Esse conjunto alimenta views de analytics e identifica gaps documentais acionáveis para curadoria da base.

---

## Otimização de latência

Pipeline inicial: aproximadamente **15 segundos** por resposta (cinco chamadas sequenciais ao modelo).

Após paralelização das tarefas independentes via `asyncio.gather` e introdução do gate condicional na reescrita: aproximadamente **10 segundos** — redução de 34% sem perda de qualidade, validada por teste comparativo em conjunto fixo de perguntas (before/after).

Próxima fronteira identificada: substituir o reranker baseado em LLM (~2,5s) por um cross-encoder local (~200ms) usando PyTorch e Transformers.

---

## Decisões técnicas

**Por que migrar de n8n para Python puro.** O projeto começou com n8n orquestrando os workflows — bom para validar a ideia, mas opaco para debug, difícil de versionar (workflows em JSON gigante) e dependente de uma máquina rodando o n8n. A migração trouxe observabilidade, testabilidade, controle fino sobre cada etapa e removeu a dependência de infraestrutura extra. A última peça migrada foi a própria ingestão de documentos, que hoje roda dentro da API hospedada.

**Por que prompt genérico em vez de regras rígidas.** Tentações iniciais de tunar o prompt para casos específicos se mostraram contraproducentes — o modelo passou a parecer travado, recitando templates fixos em vez de adaptar a resposta. A virada foi confiar no raciocínio do modelo, dando contexto rico da plataforma e instruções claras sobre adaptação (se a pergunta pede "resumido", responda em 2-4 linhas). O mesmo prompt cobre orientação, resposta-cliente e bug.

**Por que feedback como métrica de qualidade real.** O score da busca (média em torno de 0.30) reflete relevância da recuperação, não qualidade da resposta. A taxa de aprovação dos usuários reais é a verdade do sistema. O score continua útil como sinal de debug, não como métrica de produto.

**Por que separar API e bot em repositórios distintos.** Acoplamento conceitual, desacoplamento técnico. Cada um deploya independente, tem ciclo próprio, e o bot pode ser substituído por qualquer outro frontend (Slack, Teams, web) sem tocar na API.

**Por que retrieval em vez de fine-tuning.** No curto prazo, retrieval com boa documentação resolve. Fine-tuning é caro, demora, exige conjunto rotulado relevante, e a base de documentação muda. O loop de feedback gera dataset rotulado naturalmente — quando fizer sentido, virará insumo para tuning. Por enquanto, melhorias na ingestão e no chunking dão muito mais retorno.

---

## Métricas em produção

- **1.099 chunks** indexados a partir do manual oficial da plataforma (720 páginas)
- **102 interações reais**, **14 usuários distintos**
- Taxa de aprovação dos analistas: **90%** (sobre as 62 respostas avaliadas)
- Latência mediana: aproximadamente **10 segundos**
- **99% das conversas em DM privada** (analistas adotaram o canal direto)
- 6 gaps documentais identificados pelo feedback negativo, encaminhados para curadoria da base

---

## Estrutura do projeto

```
projeto_digi_python/        (este repositorio)
├── main.py                 ponto de entrada FastAPI
├── src/
│   ├── api/                endpoints (rag, ingest, history, feedback)
│   ├── pipeline/           orquestrador do fluxo RAG
│   ├── agents/             classifier, scope_validator, rag_agent, formatter
│   ├── services/           openai, supabase, history, ingestion
│   └── models/             contratos Pydantic
├── sql/                    migracoes e views de analytics
├── requirements.txt
├── squarecloud.app         configuracao de deploy
└── DOCUMENTACAO.md         documentacao tecnica detalhada
```

Bot Discord em repositório separado: [github.com/NicolasLimao/digi-bot](https://github.com/NicolasLimao/digi-bot)

---

## Deploy

Hospedado na SquareCloud com auto-deploy via GitHub: cada `git push` em `main` aciona redeploy automático da app correspondente. Variáveis sensíveis (chaves de API e tokens) ficam apenas no painel da hospedagem, nunca em arquivo versionado. O `.env` local serve apenas para desenvolvimento.

---

## Documentação técnica

Para detalhes de implementação, fluxos completos, schemas de banco e justificativas internas de cada decisão, consulte [DOCUMENTACAO.md](DOCUMENTACAO.md).
