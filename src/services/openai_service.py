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
                        "content": "You are a classifier for Digisac support questions. Classify the user query into exactly one of three categories: 'orientacao' (procedural guidance), 'resposta-cliente' (ready-to-send customer response), or 'bug' (bug report/analysis). Respond with ONLY the category name, nothing else."
                    },
                    {"role": "user", "content": query}
                ],
                temperature=0,
                max_tokens=20
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
                        "content": "You are a scope validator for Digisac support. Digisac is a customer communication platform. Determine if the user query is about Digisac or related to its use. Respond in JSON format: {\"dentro_do_escopo\": boolean, \"motivo\": string (if false, explain why)}"
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

    async def generate_response(self, query: str, chunks: List[str], mode: str) -> str:
        """Generate response using RAG context from chunks"""
        logger.info(f"[OpenAIService] Generating response for mode: {mode}, chunks: {len(chunks)}")

        if not self.client:
            logger.warning("[OpenAIService] No API key, using mock response")
            return f"[MOCK] Resposta para: {query}\nChunks usados: {len(chunks)}"

        try:
            context = "\n".join([f"- {chunk.content if hasattr(chunk, 'content') else chunk}" for chunk in chunks])

            mode_instructions = {
                "orientacao": "Provide procedural guidance in bullet points.",
                "resposta-cliente": "Provide a response ready to send to a customer via WhatsApp.",
                "bug": "Provide a bug analysis with steps to reproduce and workarounds."
            }

            system_prompt = f"""You are Digi, an AI assistant for Digisac internal support. You help N1 analysts with technical questions about the Digisac platform.

{mode_instructions.get(mode, "Provide a helpful response.")}

Use the context provided below to answer the user's question. Be concise and direct.

CONTEXT:
{context}"""

            response = await self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query}
                ],
                temperature=0.7,
                max_tokens=500
            )

            answer = response.choices[0].message.content.strip()
            logger.info(f"[OpenAIService] Generated response ({len(answer)} chars)")
            return answer

        except Exception as e:
            logger.error(f"[OpenAIService] Error generating response: {str(e)}")
            return f"Desculpe, ocorreu um erro ao gerar a resposta. Erro: {str(e)}"

    async def format_response(self, response: str, mode: str) -> str:
        """Format response based on mode (post-processing)"""
        logger.info(f"[OpenAIService] Formatting response for mode: {mode}")

        if mode == "orientacao":
            if not response.startswith("- ") and not response.startswith("•"):
                lines = response.split("\n")
                formatted = "\n".join([f"- {line.strip()}" if line.strip() and not line.startswith("-") and not line.startswith("•") else line for line in lines])
                return formatted
        elif mode == "resposta-cliente":
            return response
        elif mode == "bug":
            if "ERRO:" not in response and "ERROR:" not in response:
                return f"ERRO ENCONTRADO:\n{response}"

        return response
