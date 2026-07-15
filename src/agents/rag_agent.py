from typing import Any

from src.agents.base import Agent
from src.config import Settings, get_settings
from src.models.schemas import Document
from src.services.openai_service import OpenAIService
from src.services.supabase_service import SupabaseService


class RAGAgent(Agent):
    def __init__(
        self,
        openai_service: OpenAIService,
        supabase_service: SupabaseService,
        config: Settings | None = None,
    ):
        super().__init__("RAG")
        self.openai = openai_service
        self.supabase = supabase_service
        self.config = config or get_settings()

    def _needs_rewrite(self, query: str, history_context: str) -> bool:
        """Decide se vale reescrever a query antes de buscar (item 3: pular em pergunta simples)."""
        if history_context:
            return True  # follow-up pode ter referência a resolver
        if len(query) > 120 or "\n" in query:
            return True  # mensagem longa/ruidosa
        q = query.lower()
        return any(k in q for k in ["cliente", "obs.", "obs:"])

    async def retrieve(self, query: str, history_context: str = "", k: int = 10) -> dict[str, Any]:
        """Recuperação (não depende do modo): reescreve (condicional) -> embedding -> busca -> rerank."""
        if self._needs_rewrite(query, history_context):
            search_query = await self.openai.rewrite_query(query, history_context)
        else:
            search_query = query
            self.logger.info(f"[{self.name}] Rewrite pulado (pergunta simples)")
        self.logger.info(
            f"[{self.name}] Search query prepared",
            extra={"extras": {"query_chars": len(search_query)}},
        )

        embedding = await self.openai.get_embeddings(search_query)

        final_n = min(k, self.config.max_chunks)
        candidate_k = final_n + 5

        documents = await self.supabase.search_hybrid(
            embedding=embedding,
            query=search_query,
            k=candidate_k,
            score_threshold=self.config.score_threshold,
        )
        self.logger.info(f"[{self.name}] Retrieved {len(documents)} candidate documents")

        documents = await self.openai.rerank(search_query, documents, top_n=final_n)
        self.logger.info(f"[{self.name}] After rerank: {len(documents)} documents")

        fontes = [
            (doc.metadata.get("fonte") if isinstance(doc.metadata, dict) else None) or doc.id
            for doc in documents
        ]
        avg_score = (
            (sum(doc.score or 0.0 for doc in documents) / len(documents)) if documents else 0.0
        )

        return {
            "documents": documents,
            "search_query": search_query,
            "fontes": fontes,
            "score": avg_score,
            "chunks_used": len(documents),
        }

    async def generate(
        self,
        query: str,
        documents: list[Document],
        mode: str = "orientacao",
        history_context: str = "",
    ) -> str:
        """Geração (precisa do modo + documentos recuperados)."""
        if not documents:
            self.logger.warning(f"[{self.name}] No relevant documents found")
            response = (
                "Desculpe, não encontrei informações sobre este tópico na base de conhecimento."
            )
        else:
            response = await self.openai.generate_response(query, documents, mode, history_context)
        return await self.openai.format_response(response, mode)

    async def execute(
        self,
        query: str,
        mode: str = "orientacao",
        k: int = 10,
        history_context: str = "",
    ) -> dict[str, Any]:
        """Compat: recuperação + geração em sequência (usado fora do caminho paralelo)."""
        self.logger.info(
            "RAG execution started",
            extra={"extras": {"query_chars": len(query), "mode": mode, "k": k}},
        )

        try:
            retr = await self.retrieve(query, history_context, k)
            formatted_response = await self.generate(
                query, retr["documents"], mode, history_context
            )

            self.logger.info(f"[{self.name}] Generated response (score={retr['score']:.2f})")
            return {
                "response": formatted_response,
                "mode": mode,
                "score": retr["score"],
                "chunks_used": retr["chunks_used"],
                "documents": retr["documents"],
                "search_query": retr["search_query"],
                "fontes": retr["fontes"],
            }

        except Exception:
            self.logger.exception(f"[{self.name}] RAG execution failed")
            return {
                "response": "Não foi possível consultar a base de conhecimento agora.",
                "mode": mode,
                "score": 0.0,
                "chunks_used": 0,
                "documents": [],
                "search_query": query,
                "fontes": [],
            }
