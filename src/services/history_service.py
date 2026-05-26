from typing import Any, List, Optional
from datetime import datetime, timedelta, timezone
from src.logger import get_logger
from src.models.schemas import HistoryEntry, HistoryContextRequest

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
        processing_time_ms: int = 0,
        pergunta_reescrita: Optional[str] = None,
        fontes: Optional[List[Any]] = None,
        canal: Optional[str] = None
    ) -> Optional[str]:
        """Save a question-response pair to Supabase database"""
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
                "pergunta_reescrita": pergunta_reescrita,
                "fontes": fontes,
                "canal": canal,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }

            if not self.supabase or not hasattr(self.supabase, 'table'):
                logger.warning("[HistoryService] Supabase client not initialized, using mock")
                return f"history_{hash(user_id + pergunta) % 100000}"

            # Insert into Supabase
            response = self.supabase.table(self.table_name).insert(data).execute()

            if response.data:
                entry_id = response.data[0].get('id')
                logger.info(f"[HistoryService] Interaction saved with ID: {entry_id}")
                return str(entry_id)
            else:
                logger.error("[HistoryService] Insert returned no data")
                return None

        except Exception as e:
            logger.error(f"[HistoryService] Error saving interaction: {str(e)}")
            return None

    async def update_feedback(self, interaction_id: str, feedback: str) -> bool:
        """Update the feedback (positivo/negativo) of a saved interaction"""
        try:
            logger.info(f"[HistoryService] Updating feedback for {interaction_id}: {feedback}")

            if not self.supabase or not hasattr(self.supabase, 'table'):
                logger.warning("[HistoryService] Supabase client not initialized, skip feedback")
                return False

            response = (
                self.supabase.table(self.table_name)
                .update({"feedback": feedback})
                .eq("id", interaction_id)
                .execute()
            )

            updated = bool(response.data)
            logger.info(f"[HistoryService] Feedback updated: {updated}")
            return updated

        except Exception as e:
            logger.error(f"[HistoryService] Error updating feedback: {str(e)}")
            return False

    async def get_recent_history(self, user_id: str, limit: int = 5, within_minutes: Optional[int] = None) -> List[HistoryEntry]:
        """Fetch recent conversations for a user from Supabase, ordered by timestamp DESC.
        If within_minutes is set, only entries newer than that window are returned."""
        try:
            logger.info(f"[HistoryService] Fetching {limit} recent entries for user {user_id}")

            if not self.supabase or not hasattr(self.supabase, 'table'):
                logger.warning("[HistoryService] Supabase client not initialized, no history")
                return []

            # Query Supabase
            query = (
                self.supabase.table(self.table_name)
                .select("*")
                .eq("user_id", user_id)
            )
            if within_minutes:
                cutoff = (datetime.now(timezone.utc) - timedelta(minutes=within_minutes)).isoformat()
                query = query.gte("timestamp", cutoff)
            response = (
                query
                .order("timestamp", desc=True)
                .limit(limit)
                .execute()
            )

            history = []
            if response.data:
                for item in response.data:
                    entry = HistoryEntry(
                        id=item.get('id'),
                        user_id=item.get('user_id'),
                        pergunta=item.get('pergunta'),
                        resposta=item.get('resposta'),
                        modo=item.get('modo', 'orientacao'),
                        score=item.get('score', 0.0),
                        chunks_used=item.get('chunks_used', 0),
                        processing_time_ms=item.get('processing_time_ms', 0),
                        timestamp=item.get('timestamp')
                    )
                    history.append(entry)

            logger.info(f"[HistoryService] Retrieved {len(history)} entries")
            return history

        except Exception as e:
            logger.error(f"[HistoryService] Error fetching history: {str(e)}")
            return []

    async def format_history_for_prompt(self, user_id: str, limit: int = 5, within_minutes: Optional[int] = None) -> str:
        """Format conversation history into a string for prompt injection (chronological order)"""
        try:
            logger.info(f"[HistoryService] Formatting history for user {user_id}")

            entries = await self.get_recent_history(user_id, limit, within_minutes)

            if not entries:
                logger.info("[HistoryService] No history found, returning empty context")
                return ""

            # get_recent_history retorna do mais novo ao mais antigo; inverte para ler cronologicamente
            entries = list(reversed(entries))

            formatted = "HISTÓRICO DA CONVERSA (mensagens anteriores deste usuário, da mais antiga à mais recente):\n\n"
            for i, entry in enumerate(entries, 1):
                formatted += f"[{i}] Usuário: {entry.pergunta}\n"
                formatted += f"    Digi: {entry.resposta}\n\n"

            logger.info(f"[HistoryService] History formatted ({len(entries)} entries)")
            return formatted

        except Exception as e:
            logger.error(f"[HistoryService] Error formatting history: {str(e)}")
            return ""

    async def clear_old_history(self, days_to_keep: int = 30) -> int:
        """Delete conversation history older than N days from Supabase"""
        try:
            logger.info(f"[HistoryService] Clearing history older than {days_to_keep} days")

            if not self.supabase:
                logger.warning("[HistoryService] Supabase client not initialized")
                return 0

            # Calculate cutoff date
            cutoff_date = (datetime.now(timezone.utc) - timedelta(days=days_to_keep)).isoformat()

            # Delete from Supabase
            response = (
                self.supabase.table(self.table_name)
                .delete()
                .lt("timestamp", cutoff_date)
                .execute()
            )

            deleted_count = len(response.data) if response.data else 0
            logger.info(f"[HistoryService] Deleted {deleted_count} old entries")
            return deleted_count

        except Exception as e:
            logger.error(f"[HistoryService] Error clearing old history: {str(e)}")
            return 0
