import asyncio
from typing import Any

from supabase import create_client

from src.logger import get_logger
from src.models.schemas import Document

logger = get_logger(__name__)


class SupabaseServiceError(RuntimeError):
    pass


class SupabaseService:
    def __init__(self, url: str = "", key: str = "", client: Any = None):
        self.url = url
        self.client = client
        if self.client is None and url and key:
            try:
                self.client = create_client(url, key)
            except Exception as exc:
                raise SupabaseServiceError("Could not initialize Supabase client") from exc

    def _require_client(self) -> Any:
        if self.client is None:
            raise SupabaseServiceError("Supabase is not configured")
        return self.client

    async def save_document(
        self,
        content: str,
        embedding: list[float],
        metadata: dict[str, Any],
    ) -> str | None:
        client = self._require_client()
        data = {
            "content": content,
            "embedding": embedding,
            "metadata": metadata,
            "fonte": metadata.get("fonte", "ingestion"),
        }

        try:
            response = await asyncio.to_thread(
                lambda: client.table("documents").insert(data).execute()
            )
        except Exception as exc:
            logger.exception("Supabase document insert failed")
            raise SupabaseServiceError("Document storage failed") from exc
        return str(response.data[0]["id"]) if response.data else None

    async def search_hybrid(
        self,
        embedding: list[float],
        query: str,
        k: int = 10,
        score_threshold: float = 0.0,
    ) -> list[Document]:
        client = self._require_client()
        params = {
            "query_text": query,
            "query_embedding": str(embedding),
            "match_count": k,
            "full_text_weight": 0.5,
            "semantic_weight": 0.5,
        }
        try:
            response = await asyncio.to_thread(
                lambda: client.rpc("match_documents_hybrid", params).execute()
            )
        except Exception as exc:
            logger.exception("Supabase hybrid search failed")
            raise SupabaseServiceError("Knowledge-base search failed") from exc

        documents: list[Document] = []
        for index, item in enumerate(response.data or []):
            score = float(item.get("score") or 0.0)
            if score < score_threshold:
                continue
            documents.append(
                Document(
                    id=str(item.get("id", f"chunk_{index}")),
                    content=str(item.get("content") or ""),
                    embedding=item.get("embedding") or embedding,
                    metadata=item.get("metadata") or {},
                    score=min(max(score, 0.0), 1.0),
                )
            )
        logger.info(
            "Hybrid search completed",
            extra={"extras": {"candidates": len(response.data or []), "accepted": len(documents)}},
        )
        return documents

    async def get_document(self, doc_id: str) -> Document | None:
        client = self._require_client()
        try:
            response = await asyncio.to_thread(
                lambda: client.table("documents").select("*").eq("id", doc_id).execute()
            )
        except Exception as exc:
            logger.exception("Supabase document read failed")
            raise SupabaseServiceError("Document read failed") from exc
        if not response.data:
            return None
        item = response.data[0]
        return Document(
            id=str(item["id"]),
            content=str(item.get("content") or ""),
            embedding=item.get("embedding") or [],
            metadata=item.get("metadata") or {},
            score=item.get("score"),
        )
