import asyncio
from typing import Optional
from time import time
from src.agents.classifier import ClassifierAgent
from src.agents.scope_validator import ScopeValidatorAgent
from src.agents.rag_agent import RAGAgent
from src.agents.formatter_agent import FormatterAgent
from src.services.history_service import HistoryService
from src.models.schemas import QueryResponse
from src.config import settings
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
        mode: Optional[str] = None,
        canal: str = "desconhecido"
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
            # Histórico primeiro (a reescrita da query o utiliza)
            history_context = ""
            if settings.history_enabled:
                history_context = await self.history.format_history_for_prompt(
                    user_id, limit=4, within_minutes=60
                )
                if history_context:
                    self.logger.info(f"[RAGPipeline] Injecting conversation history for user {user_id}")

            # Classificar + validar + recuperar — as três EM PARALELO
            # (classificador dedicado é mais preciso; paralelizar esconde o custo sob a recuperação)
            classify_task = asyncio.create_task(self.classifier.execute(query))
            validate_task = asyncio.create_task(self.validator.execute(query))
            retrieve_task = asyncio.create_task(self.rag_agent.retrieve(query, history_context))
            classification_raw, validation, retr = await asyncio.gather(classify_task, validate_task, retrieve_task)

            classification = mode or classification_raw
            self.logger.info(f"[RAGPipeline] modo={classification}, escopo={validation.get('dentro_do_escopo', True)}")

            # Fora do escopo: retorna sem gerar
            if not validation.get("dentro_do_escopo", True):
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

            # Geração (precisa do modo + documentos recuperados)
            formatted = await self.rag_agent.generate(
                query, retr["documents"], classification, history_context
            )

            processing_time_ms = int((time() - start_time) * 1000)

            response = QueryResponse(
                response=formatted,
                mode=classification,
                score=retr.get("score", 0.0),
                chunks_used=retr.get("chunks_used", 0),
                processing_time_ms=processing_time_ms
            )

            interaction_id = await self.history.save_interaction(
                user_id=user_id,
                pergunta=query,
                resposta=formatted,
                modo=classification,
                score=response.score,
                chunks_used=response.chunks_used,
                processing_time_ms=processing_time_ms,
                pergunta_reescrita=retr.get("search_query"),
                fontes=retr.get("fontes"),
                canal=canal
            )
            response.interaction_id = interaction_id

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
