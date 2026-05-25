from fastapi import APIRouter, HTTPException, Depends
from src.models.schemas import QueryRequest, QueryResponse
from src.services.openai_service import OpenAIService
from src.services.supabase_service import SupabaseService
from src.services.history_service import HistoryService
from src.agents.classifier import ClassifierAgent
from src.agents.scope_validator import ScopeValidatorAgent
from src.agents.rag_agent import RAGAgent
from src.agents.formatter_agent import FormatterAgent
from src.pipeline.rag_pipeline import RAGPipeline
from src.config import settings
from src.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/api/rag", tags=["rag"])


def get_openai_service() -> OpenAIService:
    """Get OpenAIService instance with real client"""
    return OpenAIService(api_key=settings.openai_api_key)


def get_supabase_service() -> SupabaseService:
    """Get SupabaseService instance with real client"""
    return SupabaseService(url=settings.supabase_url, key=settings.supabase_anon_key)


def get_history_service() -> HistoryService:
    """Get HistoryService with a real Supabase client"""
    svc = SupabaseService(url=settings.supabase_url, key=settings.supabase_anon_key)
    return HistoryService(svc.client)


def get_rag_pipeline(
    openai: OpenAIService = Depends(get_openai_service),
    supabase: SupabaseService = Depends(get_supabase_service),
    history: HistoryService = Depends(get_history_service)
) -> RAGPipeline:
    """Get RAGPipeline with all dependencies"""
    classifier = ClassifierAgent(openai)
    validator = ScopeValidatorAgent(openai)
    rag_agent = RAGAgent(openai, supabase)
    formatter = FormatterAgent()

    return RAGPipeline(classifier, validator, rag_agent, formatter, history)


@router.post("/query", response_model=QueryResponse)
async def query_rag(
    request: QueryRequest,
    user_id: str,
    pipeline: RAGPipeline = Depends(get_rag_pipeline)
) -> QueryResponse:
    """
    Process a query through the complete RAG pipeline.

    Request:
    - query: User's question
    - mode: Optional mode (orientacao, resposta-cliente, bug) - auto-classified if not provided
    - user_id: User ID for history tracking

    Response:
    - response: Generated response
    - mode: Classification mode
    - score: Average relevance score
    - chunks_used: Number of KB chunks used
    - processing_time_ms: Time to process
    """
    logger.info(f"[API] RAG query from user {user_id}: {request.query[:50]}...")

    try:
        if not user_id or len(user_id) == 0:
            raise HTTPException(status_code=400, detail="user_id is required")

        if not request.query or len(request.query.strip()) == 0:
            raise HTTPException(status_code=400, detail="query cannot be empty")

        result = await pipeline.process(
            query=request.query,
            user_id=user_id,
            mode=request.mode
        )

        logger.info(f"[API] Query processed successfully (score={result.score:.2f})")
        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[API] Error processing RAG query: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")
