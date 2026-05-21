from typing import Optional
from time import time
from src.agents.classifier import ClassifierAgent
from src.agents.scope_validator import ScopeValidatorAgent
from src.agents.rag_agent import RAGAgent
from src.agents.formatter_agent import FormatterAgent
from src.services.history_service import HistoryService
from src.models.schemas import QueryResponse
from src.logger import get_logger

logger = get_logger(__name__)


class RAGPipeline:
    def __init__(
        self,
        classifier: ClassifierAgent,
        validator: ScopeValidatorAgent,
        rag_agent: RAGAgent,
        formatter: FormatterAgent,
        history_service: HistoryService
    ):
        self.classifier = classifier
        self.validator = validator
        self.rag_agent = rag_agent
        self.formatter = formatter
        self.history = history_service
        self.logger = logger

    async def process(
        self,
        query: str,
        user_id: str,
        mode: Optional[str] = None
    ) -> QueryResponse:
        """
        Execute full RAG pipeline:
        1. Classify query
        2. Validate scope
        3. Run RAG if in scope
        4. Format response
        5. Save to history
        6. Return response
        """
        start_time = time()
        self.logger.info(f"[RAGPipeline] Starting pipeline for user {user_id}")

        try:
            if not mode:
                classification = await self.classifier.execute(query)
                self.logger.info(f"[RAGPipeline] Classified as: {classification}")
            else:
                classification = mode

            validation = await self.validator.execute(query)
            self.logger.info(f"[RAGPipeline] Scope validation: {validation['dentro_do_escopo']}")

            if not validation.get("dentro_do_escopo", False):
                response_text = f"Desculpe, sua pergunta está fora do escopo de suporte. Motivo: {validation.get('motivo', 'Não especificado')}"
                response = QueryResponse(
                    response=response_text,
                    mode="orientacao",
                    score=0.0,
                    chunks_used=0,
                    processing_time_ms=int((time() - start_time) * 1000)
                )
                self.logger.info("[RAGPipeline] Query out of scope, returning early")
                return response

            rag_result = await self.rag_agent.execute(query, mode=classification)
            formatted = await self.formatter.execute(rag_result["response"], classification)

            processing_time_ms = int((time() - start_time) * 1000)

            response = QueryResponse(
                response=formatted,
                mode=classification,
                score=rag_result.get("score", 0.0),
                chunks_used=rag_result.get("chunks_used", 0),
                processing_time_ms=processing_time_ms
            )

            await self.history.save_interaction(
                user_id=user_id,
                pergunta=query,
                resposta=formatted,
                modo=classification,
                score=response.score,
                chunks_used=response.chunks_used,
                processing_time_ms=processing_time_ms
            )

            self.logger.info(f"[RAGPipeline] Completed pipeline in {processing_time_ms}ms")
            return response

        except Exception as e:
            self.logger.error(f"[RAGPipeline] Error in pipeline: {str(e)}")
            processing_time_ms = int((time() - start_time) * 1000)

            error_response = QueryResponse(
                response=f"Erro ao processar pergunta: {str(e)}",
                mode="orientacao",
                score=0.0,
                chunks_used=0,
                processing_time_ms=processing_time_ms
            )

            await self.history.save_interaction(
                user_id=user_id,
                pergunta=query,
                resposta=error_response.response,
                modo="orientacao",
                score=0.0,
                chunks_used=0,
                processing_time_ms=processing_time_ms
            )

            return error_response
