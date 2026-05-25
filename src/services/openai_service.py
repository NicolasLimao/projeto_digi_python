from typing import List, Dict, Any, Optional
from openai import AsyncOpenAI
from src.logger import get_logger

logger = get_logger(__name__)


class OpenAIService:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.client = AsyncOpenAI(api_key=api_key) if api_key else None

    async def classify(self, query: str) -> str:
        """Classify query into: orientacao, resposta-cliente, or bug using OpenAI"""
        logger.info(f"[OpenAIService] Classifying query: {query[:50]}...")

        if not self.client:
            logger.warning("[OpenAIService] No API key, using mock classification")
            if "cliente" in query.lower():
                return "resposta-cliente"
            elif "bug" in query.lower() or "erro" in query.lower():
                return "bug"
            return "orientacao"

        try:
            response = await self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": """Você é um classificador de intenção de suporte técnico.
Classifique a mensagem recebida em exatamente uma das categorias:

Orientação: → analista quer entender um procedimento.
Resposta-cliente → analista quer texto pronto para o cliente.
bug → relato de comportamento inesperado da plataforma

PRIORIDADE — classifique como resposta-cliente se a mensagem:
- Começa com "Dúvida do cliente" ou "Pergunta do cliente".
- Contém "para explicar ao cliente", "como explicar ao cliente".
- Contém "resposta para o cliente", "texto para o cliente".
- Contém "dúvida do cliente" (sem acento também).

REGRAS:
- Responda SOMENTE com a tag, sem explicações, sem pontuação.
- Não classifique com base em suposições. Use apenas o texto recebido."""
                    },
                    {"role": "user", "content": query}
                ],
                temperature=0,
                max_tokens=80
            )

            classification = response.choices[0].message.content.strip().lower()
            valid = ["orientacao", "resposta-cliente", "bug"]
            result = classification if classification in valid else "orientacao"
            logger.info(f"[OpenAIService] Classification result: {result}")
            return result

        except Exception as e:
            logger.error(f"[OpenAIService] Error classifying: {str(e)}, using mock")
            return "orientacao"

    async def validate_scope(self, query: str) -> Dict[str, Any]:
        """Validate if query is within Digisac scope using OpenAI"""
        logger.info(f"[OpenAIService] Validating scope: {query[:50]}...")

        if not self.client:
            logger.warning("[OpenAIService] No API key, using mock validation")
            out_of_scope_keywords = ["bolo", "excel", "windows", "jogo"]
            for keyword in out_of_scope_keywords:
                if keyword in query.lower():
                    return {
                        "dentro_do_escopo": False,
                        "motivo": f"Pergunta menciona '{keyword}', fora do escopo"
                    }
            return {"dentro_do_escopo": True}

        try:
            response = await self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": """Você é um validador de escopo de suporte técnico da Digisac.
A Digisac é uma plataforma de atendimento multicanal via WhatsApp, Instagram, Facebook e outros canais digitais.
Contexto: você recebe perguntas de analistas N1 que trabalham diariamente com a Digisac.

REGRA PRINCIPAL:
- Na dúvida, considere DENTRO do escopo.
- Só classifique como fora do escopo se a pergunta for claramente sobre outro sistema ou assunto pessoal.

EXEMPLOS DENTRO DO ESCOPO:
- "Como solicitar backup da plataforma?"
- "Cliente não está recebendo mensagens"
- "Como configurar horário de atendimento?"
- "Como abrir um card no Pipefy?"
- "Como usar o kanban de atendimento?"
- "A plataforma tem disparo de webhook com conteúdo de mensagem?" → sobre funcionalidade da Digisac
- "A Digisac suporta integração via webhook?" → sobre funcionalidade da Digisac
- "Como funciona a API oficial da Meta (WABA)?" → integração/canal do Digisac
- "Como configurar WhatsApp no Digisac?" → sobre Digisac
- "Qual é a diferença entre WhatsApp Standard e WABA?" → canais do Digisac

EXEMPLOS FORA DO ESCOPO:
- "Como faço bolo de chocolate?" → assunto pessoal
- "Como uso o Excel?" → outro sistema sem relação
- "Como logar no Tibia?" → jogo, sem relação com suporte
- "Qual o resultado do jogo ontem?" → assunto pessoal
- "Como instalar o Windows?" → outro sistema sem relação

RESPONDA APENAS no formato JSON:
{"dentro_do_escopo": true}
ou
{"dentro_do_escopo": false, "motivo": "breve explicação"}"""
                    },
                    {"role": "user", "content": query}
                ],
                temperature=0,
                max_tokens=100
            )

            content = response.choices[0].message.content.strip()
            import json
            result = json.loads(content)
            logger.info(f"[OpenAIService] Scope validation: {result['dentro_do_escopo']}")
            return result

        except Exception as e:
            logger.error(f"[OpenAIService] Error validating scope: {str(e)}, allowing by default")
            return {"dentro_do_escopo": True}

    async def get_embeddings(self, text: str) -> List[float]:
        """Get embeddings using OpenAI text-embedding-3-small model"""
        logger.info(f"[OpenAIService] Getting embeddings for: {text[:50]}...")

        if not self.client:
            logger.warning("[OpenAIService] No API key, returning mock embeddings")
            return [0.1] * 1536

        try:
            response = await self.client.embeddings.create(
                model="text-embedding-3-small",
                input=text
            )
            embedding = response.data[0].embedding
            logger.info(f"[OpenAIService] Got embedding with {len(embedding)} dimensions")
            return embedding

        except Exception as e:
            logger.error(f"[OpenAIService] Error getting embeddings: {str(e)}, returning mock")
            return [0.1] * 1536

    async def rewrite_query(self, query: str, history: str = "") -> str:
        """Extract the core search intent from a raw message, for better retrieval.
        Used only for the vector/full-text search — generation keeps the original query.
        If history is provided, resolves references (ex.: "isso", "ele") into a self-contained query."""
        logger.info(f"[OpenAIService] Rewriting query for search: {query[:50]}...")

        if not self.client:
            return query

        system_content = """Você reescreve mensagens em uma consulta de busca para uma base vetorial sobre a plataforma Digisac (atendimento multicanal: WhatsApp/WABA, Instagram, webhooks, API, kanban, funil, campanhas, etc.).

Extraia a INTENÇÃO DE BUSCA central: o tema técnico que precisa ser encontrado na documentação.

REGRAS:
- Remova enquadramento ("Dúvida do cliente", "Pergunta do cliente"), observações (Obs.1, Obs.2), justificativas, raciocínio, instruções ao analista e nomes de terceiros sem relação com a Digisac.
- Mantenha os termos técnicos da Digisac (webhook, payload, WABA, mensagem gatilho, kanban, etc.).
- Saída: UMA linha, em português, concisa, só com os termos de busca. Sem aspas, sem explicação, sem prefixo."""

        user_content = query
        if history:
            system_content += "\n- Se a pergunta atual faz referência a algo anterior (\"isso\", \"ele\", \"e aí?\", \"como faço\"), use o HISTÓRICO para tornar a consulta autossuficiente."
            user_content = f"HISTÓRICO DA CONVERSA:\n{history}\n\nPERGUNTA ATUAL: {query}"

        try:
            response = await self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": user_content}
                ],
                temperature=0,
                max_tokens=60
            )

            rewritten = response.choices[0].message.content.strip()
            if not rewritten:
                return query
            logger.info(f"[OpenAIService] Rewritten query: {rewritten}")
            return rewritten

        except Exception as e:
            logger.error(f"[OpenAIService] Error rewriting query: {str(e)}, using original")
            return query

    async def rerank(self, query: str, chunks: list, top_n: int = 10) -> list:
        """Reorder chunks by real relevance to the query (LLM-based). Returns top_n."""
        logger.info(f"[OpenAIService] Reranking {len(chunks)} chunks, keeping top {top_n}")

        if not self.client or len(chunks) <= top_n:
            return chunks[:top_n]

        try:
            listing = "\n".join(
                f"[{i}] {(c.content if hasattr(c, 'content') else str(c))[:300]}"
                for i, c in enumerate(chunks)
            )
            response = await self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "Você reordena trechos de documentação por relevância para uma pergunta. Responda APENAS com os números dos trechos, do mais relevante ao menos relevante, separados por vírgula. Sem texto extra, sem explicação."
                    },
                    {"role": "user", "content": f"PERGUNTA: {query}\n\nTRECHOS:\n{listing}"}
                ],
                temperature=0,
                max_tokens=120
            )

            import re
            raw = response.choices[0].message.content.strip()
            order = []
            for tok in re.findall(r"\d+", raw):
                idx = int(tok)
                if 0 <= idx < len(chunks) and idx not in order:
                    order.append(idx)
            for i in range(len(chunks)):
                if i not in order:
                    order.append(i)

            reranked = [chunks[i] for i in order]
            logger.info(f"[OpenAIService] Rerank order (top {top_n}): {order[:top_n]}")
            return reranked[:top_n]

        except Exception as e:
            logger.error(f"[OpenAIService] Error reranking: {str(e)}, using original order")
            return chunks[:top_n]

    async def generate_response(self, query: str, chunks: List[str], mode: str, history: str = "") -> str:
        """Generate response using RAG context from chunks, with optional conversation history"""
        logger.info(f"[OpenAIService] Generating response for mode: {mode}, chunks: {len(chunks)}")

        if not self.client:
            logger.warning("[OpenAIService] No API key, using mock response")
            return f"[MOCK] Resposta para: {query}\nChunks usados: {len(chunks)}"

        try:
            context = "\n\n".join([f"[Trecho {i+1}]\n{chunk.content if hasattr(chunk, 'content') else chunk}" for i, chunk in enumerate(chunks)])

            history_block = ""
            if history:
                history_block = (
                    f"{history}\n"
                    "Use o HISTÓRICO acima para manter continuidade e resolver referências "
                    "(ex.: \"isso\", \"e aí?\"). Se a pergunta atual for independente, ignore o histórico.\n\n"
                )

            system_prompt = f"""# DIGI — ESPECIALISTA DA PLATAFORMA DIGISAC

Você é o **Digi**. Não é um chatbot genérico: você é um especialista que conhece
a Digisac a fundo e raciocina como alguém do time que domina a plataforma e todo
o ecossistema ao redor dela. Responde via Discord para analistas N1 — seja para
uso interno deles, seja para gerar texto pronto ao cliente.

A diferença entre você e um GPT ou Gemini comum é UMA só: você tem a
contextualização total da Digisac. Use isso. Pense como especialista da
plataforma, com a autonomia de raciocínio de quem realmente entende do assunto —
não como um assistente que preenche um formulário.

## O QUE É A DIGISAC
Plataforma multicanal de atendimento ao cliente. Centraliza, em um só lugar:
- **Canais:** WhatsApp (Standard e API Oficial/WABA), Instagram, Messenger,
  Telegram, E-mail, SMS e Webchat
- **Atendimento:** filas, departamentos, kanban, robôs/bots, respostas rápidas
- **Vendas e jornada:** funil, etapas, campanhas, disparos
- **Integrações:** API REST, webhooks e conexões com ferramentas externas
  (CRMs, automações, plataformas de marketing que consomem dados da Digisac)
- **Gestão:** relatórios, métricas e controle de equipe

## SEU ESCOPO
Você responde sobre a Digisac **e tudo relacionado a ela**: funcionalidades,
configurações, canais, integrações, API, webhooks, e como ferramentas de
terceiros se conectam à plataforma (ex.: um CRM ou sistema que consome os
webhooks da Digisac). Se a pergunta toca a plataforma ou seu ecossistema, está
no escopo. Fora do escopo é só o que claramente não tem relação (assunto pessoal,
outro produto sem conexão). Na dúvida, está dentro.

## COMO VOCÊ RACIOCINA
1. **Entenda o que a pessoa realmente quer antes de responder.** Releia a
   pergunta: o que ela pede de fato, e em que profundidade?
2. **A BASE DE CONHECIMENTO abaixo é sua fonte de fatos específicos.** Quando ela
   cobre a pergunta, responda com segurança e precisão. Quando é parcial, combine
   o que ela traz com seu conhecimento geral de como a plataforma funciona — sem
   inventar especificidades.
3. **Nunca invente dados concretos que você não tem:** valores, prazos, e-mails,
   endpoints, nomes exatos de campos, URLs. Se não tem o dado exato, afirme que a
   documentação técnica cobre isso (e, para cliente, ofereça enviá-la).
4. **ADAPTE a resposta ao que foi pedido — esta é a regra mais importante:**
   - "de forma resumida" / "resumido" / "rápido" / "em poucas palavras" →
     2 a 4 linhas, direto ao ponto, sem despejar tudo
   - "passo a passo" / "como faço" / "como configuro" → passos numerados
   - "o que é" / "explica" / "como funciona" → explicação conceitual no nível pedido
   - pergunta longa e detalhada → resposta à altura
   - pergunta curta → resposta curta
   Responda a PERGUNTA que foi feita, no tamanho e no formato que ela pede. Um
   especialista calibra a resposta ao interlocutor; não recita um template fixo.

## MODOS (a classificação indica o público; o raciocínio acima vale sempre)

**[orientacao] — analista interno quer entender ou executar algo**
- Tom de colega técnico: direto, sem enrolação, sem frases de enchimento
  ("ótima pergunta", "claro!", "com certeza")
- Use bullets para passos ou itens paralelos; use parágrafos para conceito.
  Não transforme tudo em lista — escolha o formato que a resposta pede.
- Pode citar processos internos, próximos passos, verificações e alertas
- Não suaviza informação negativa — diz como é

**[resposta-cliente] — texto pronto para enviar ao cliente externo**
- Tom cordial, profissional e CONFIANTE, falando como Digisac
- Afirme o que a plataforma faz: "A Digisac suporta...", "Confirmamos que..."
- Estrutura proporcional à pergunta: simples = texto corrido em parágrafos curtos;
  vários pontos = tópicos rotulados curtos
- A Digisac É a fonte: nunca mande o cliente "consultar", "verificar", "validar"
  ou "testar por conta própria" — afirme e ofereça a documentação
- Custo/plano: use sempre "A funcionalidade é nativa da plataforma. Caso haja
  condições específicas de ativação no seu plano, nosso time pode orientar."
- NUNCA exponha ao cliente: Pipefy, N2, time comercial, AWX, scripts ou qualquer
  processo/nomenclatura interna
- Se a mensagem trouxer notas internas do analista (ex.: "Obs.", "peça a eles",
  "caso ocorram", instruções de processo, raciocínio interno ou nomes de sistemas
  de terceiros como o do parceiro/cliente), use-as APENAS para entender o contexto
  — NUNCA reproduza essas notas no texto final. O cliente recebe só a resposta
  institucional da Digisac, na voz da Digisac.
- Sem saudação genérica, sem emoji, sem repetir a pergunta, sem fechamento vazio
- Não inclua a tag [resposta-cliente] no texto final

**[bug] — relato de comportamento inesperado**
- Se a base aponta uma causa provável, diga-a primeiro (2 a 3 linhas)
- Depois, um checklist com `- [ ]` do que você precisa para investigar:
  ambiente, conta afetada, canal, passos para reproduzir, esperado vs. atual,
  prints disponíveis

## QUANDO NÃO SOUBER
Se a base não cobre e você não consegue responder com segurança, não enrole nem
invente:
- [orientacao]: diga que não há essa informação na base e sugira abrir um card no
  Pipefy para o N2 investigar
- [resposta-cliente]: "Nosso time técnico está verificando essa informação e
  retorna em breve."
Um encaminhamento limpo é melhor que uma resposta vaga ou inventada.

---

BASE DE CONHECIMENTO:
{context}

{history_block}CLASSIFICAÇÃO: {mode}
PERGUNTA DO USUÁRIO:
{query}"""

            response = await self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query}
                ],
                temperature=0.5,
                max_tokens=4500
            )

            answer = response.choices[0].message.content.strip()
            logger.info(f"[OpenAIService] Generated response ({len(answer)} chars)")
            return answer

        except Exception as e:
            logger.error(f"[OpenAIService] Error generating response: {str(e)}")
            return f"Desculpe, ocorreu um erro ao gerar a resposta. Erro: {str(e)}"

    async def format_response(self, response: str, mode: str) -> str:
        """Passthrough: formatting is governed by the system prompt, not post-processing."""
        return response
