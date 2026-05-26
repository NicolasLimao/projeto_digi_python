from typing import Dict, Any, List
from src.agents.base import Agent
from src.services.openai_service import OpenAIService
from src.services.supabase_service import SupabaseService
from src.models.schemas import Document
from src.config import settings


class RAGAgent(Agent):
    def __init__(self, openai_service: OpenAIService, supabase_service: SupabaseService):
        super().__init__("RAG")
        self.openai = openai_service
        self.supabase = supabase_service

    async def execute(self, query: str, mode: str = "orientacao", k: int = 10, history_context: str = "") -> Dict[str, Any]:
        """
        Execute RAG pipeline:
        1. Get embeddings for query
        2. Search Supabase for similar documents
        3. Filter by score threshold
        4. Generate response with context
        5. Return response with metadata
        """
        self.logger.info(f"[{self.name}] Processing query: {query[:50]}... (mode={mode}, k={k})")

        try:
            search_query = await self.openai.rewrite_query(query, history_context)
            self.logger.info(f"[{self.name}] Search query: {search_query[:80]}")

            embedding = await self.openai.get_embeddings(search_query)
            self.logger.info(f"[{self.name}] Got embeddings ({len(embedding)} dims)")

            final_n = min(k, settings.max_chunks)
            candidate_k = final_n + 5

            documents = await self.supabase.search_hybrid(
                embedding=embedding,
                query=search_query,
                k=candidate_k,
                score_threshold=settings.score_threshold
            )

            self.logger.info(f"[{self.name}] Retrieved {len(documents)} candidate documents")

            documents = await self.openai.rerank(search_query, documents, top_n=final_n)
            self.logger.info(f"[{self.name}] After rerank: {len(documents)} documents")

            chunks = [doc.content for doc in documents]
            if not chunks:
                self.logger.warning(f"[{self.name}] No relevant documents found")
                response = "Desculpe, não encontrei informações sobre este tópico na base de conhecimento."
                avg_score = 0.0
            else:
                response = await self.openai.generate_response(query, documents, mode, history_context)
                avg_score = sum([doc.score or 0.0 for doc in documents]) / len(documents)

            formatted_response = await self.openai.format_response(response, mode)

            fontes = [
                (doc.metadata.get("fonte") if isinstance(doc.metadata, dict) else None) or doc.id
                for doc in documents
            ]

            result = {
                "response": formatted_response,
                "mode": mode,
                "score": avg_score,
                "chunks_used": len(documents),
                "documents": documents,
                "search_query": search_query,
                "fontes": fontes
            }

            self.logger.info(f"[{self.name}] Generated response (score={avg_score:.2f})")
            return result

        except Exception as e:
            self.logger.error(f"[{self.name}] Error during RAG execution: {str(e)}")
            return {
                "response": f"Erro ao processar pergunta: {str(e)}",
                "mode": mode,
                "score": 0.0,
                "chunks_used": 0,
                "documents": [],
                "search_query": query,
                "fontes": []
            }
