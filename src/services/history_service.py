from typing import Any, List, Optional
from datetime import datetime
from src.logger import get_logger
from src.models.schemas import HistoryEntry, HistoryContextRequest
import json

logger = get_logger(__name__)


class HistoryService:
    def __init__(self, supabase_client: Any):
        self.supabase = supabase_client
        self.table_name = "historico_digi"

    async def save_interaction(
        self,
        user_id: str,
        pergunta: str,
        resposta: str,
        modo: str = "orientacao",
        score: float = 0.0,
        chunks_used: int = 0,
        processing_time_ms: int = 0
    ) -> Optional[str]:
        """Save a question-response pair to database"""
        try:
            logger.info(f"[HistoryService] Saving interaction for user {user_id}")

            data = {
                "user_id": user_id,
                "pergunta": pergunta,
                "resposta": resposta,
                "modo": modo,
                "score": score,
                "chunks_used": chunks_used,
                "processing_time_ms": processing_time_ms,
                "timestamp": datetime.utcnow().isoformat()
            }

            # Mock implementation - replace with actual Supabase call
            entry_id = f"history_{hash(user_id + pergunta) % 100000}"
            logger.info(f"[HistoryService] Interaction saved with ID: {entry_id}")

            return entry_id
        except Exception as e:
            logger.error(f"[HistoryService] Error saving interaction: {str(e)}")
            return None

    async def get_recent_history(self, user_id: str, limit: int = 5) -> List[HistoryEntry]:
        """Fetch recent conversations for a user, ordered by timestamp DESC"""
        try:
            logger.info(f"[HistoryService] Fetching {limit} recent entries for user {user_id}")

            # Mock implementation - replace with actual Supabase query:
            # SELECT * FROM historico_digi
            # WHERE user_id = $1
            # ORDER BY timestamp DESC
            # LIMIT $2

            mock_history = [
                HistoryEntry(
                    id=f"entry_{i}",
                    user_id=user_id,
                    pergunta=f"Pergunta anterior {i}",
                    resposta=f"Resposta anterior {i}",
                    modo="orientacao",
                    score=0.85,
                    chunks_used=3,
                    processing_time_ms=500,
                    timestamp=datetime.utcnow().isoformat()
                )
                for i in range(min(limit, 3))
            ]

            logger.info(f"[HistoryService] Retrieved {len(mock_history)} entries")
            return mock_history
        except Exception as e:
            logger.error(f"[HistoryService] Error fetching history: {str(e)}")
            return []

    async def format_history_for_prompt(self, user_id: str, limit: int = 5) -> str:
        """Format conversation history into a string for prompt injection"""
        try:
            logger.info(f"[HistoryService] Formatting history for user {user_id}")

            entries = await self.get_recent_history(user_id, limit)

            if not entries:
                logger.info("[HistoryService] No history found, returning empty context")
                return ""

            # Format as: [1] Pergunta: ... | Resposta: ...
            formatted = "HISTÓRICO RECENTE DO ANALISTA:\n\n"
            for i, entry in enumerate(entries, 1):
                formatted += f"[{i}] Pergunta: {entry.pergunta}\n"
                formatted += f"    Resposta: {entry.resposta}\n\n"

            logger.info(f"[HistoryService] History formatted ({len(entries)} entries)")
            return formatted
        except Exception as e:
            logger.error(f"[HistoryService] Error formatting history: {str(e)}")
            return ""

    async def clear_old_history(self, days_to_keep: int = 30) -> int:
        """Delete conversation history older than N days"""
        try:
            logger.info(f"[HistoryService] Clearing history older than {days_to_keep} days")

            # Mock implementation - replace with:
            # DELETE FROM historico_digi
            # WHERE timestamp < NOW() - INTERVAL '{days_to_keep} days'

            deleted_count = 0
            logger.info(f"[HistoryService] Deleted {deleted_count} old entries")
            return deleted_count
        except Exception as e:
            logger.error(f"[HistoryService] Error clearing old history: {str(e)}")
            return 0
