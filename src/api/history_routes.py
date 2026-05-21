from fastapi import APIRouter, HTTPException, Depends
from src.services.history_service import HistoryService
from src.models.schemas import HistoryEntry, HistoryContextRequest, HistoryContextResponse
from src.logger import get_logger
from typing import List

logger = get_logger(__name__)

router = APIRouter(prefix="/api/history", tags=["history"])


def get_history_service() -> HistoryService:
    """Get HistoryService instance - replace with DI in production"""
    return HistoryService(None)


@router.post("/save")
async def save_interaction(
    user_id: str,
    pergunta: str,
    resposta: str,
    modo: str = "orientacao",
    score: float = 0.0,
    chunks_used: int = 0,
    processing_time_ms: int = 0,
    service: HistoryService = Depends(get_history_service)
):
    """Save a question-response pair to history"""
    logger.info(f"[API] Saving interaction for user {user_id}")

    try:
        entry_id = await service.save_interaction(
            user_id=user_id,
            pergunta=pergunta,
            resposta=resposta,
            modo=modo,
            score=score,
            chunks_used=chunks_used,
            processing_time_ms=processing_time_ms
        )

        if not entry_id:
            raise HTTPException(status_code=500, detail="Failed to save interaction")

        return {"id": entry_id, "status": "saved"}
    except Exception as e:
        logger.error(f"[API] Error saving interaction: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/context")
async def get_history_context(
    request: HistoryContextRequest,
    service: HistoryService = Depends(get_history_service)
) -> HistoryContextResponse:
    """Get formatted conversation history for prompt injection"""
    logger.info(f"[API] Fetching context for user {request.user_id}")

    try:
        formatted = await service.format_history_for_prompt(
            user_id=request.user_id,
            limit=request.limit
        )

        history = await service.get_recent_history(
            user_id=request.user_id,
            limit=request.limit
        )

        oldest = history[-1].timestamp if history else None

        return HistoryContextResponse(
            formatted_context=formatted,
            entry_count=len(history),
            oldest_timestamp=oldest
        )
    except Exception as e:
        logger.error(f"[API] Error fetching context: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/user/{user_id}")
async def get_user_history(
    user_id: str,
    limit: int = 10,
    service: HistoryService = Depends(get_history_service)
) -> List[HistoryEntry]:
    """Get raw conversation history for a user"""
    logger.info(f"[API] Fetching history for user {user_id} (limit={limit})")

    try:
        history = await service.get_recent_history(user_id=user_id, limit=limit)
        return history
    except Exception as e:
        logger.error(f"[API] Error fetching user history: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/cleanup")
async def cleanup_old_history(
    days_to_keep: int = 30,
    service: HistoryService = Depends(get_history_service)
):
    """Delete conversation history older than N days"""
    logger.info(f"[API] Cleaning up history older than {days_to_keep} days")

    try:
        deleted = await service.clear_old_history(days_to_keep=days_to_keep)
        return {"deleted_count": deleted, "status": "completed"}
    except Exception as e:
        logger.error(f"[API] Error cleaning up history: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
