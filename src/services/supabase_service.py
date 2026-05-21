from typing import List, Dict, Any, Optional
from supabase import create_client
from src.logger import get_logger
from src.models.schemas import Document
import json

logger = get_logger(__name__)


class SupabaseService:
    def __init__(self, url: str, key: str):
        self.url = url
        self.key = key
        try:
            self.client = create_client(url, key) if url and key else None
        except Exception as e:
            logger.warning(f"[SupabaseService] Failed to create Supabase client: {str(e)}")
            self.client = None

    async def save_document(self, content: str, embedding: List[float], metadata: Dict[str, Any]) -> str:
        """Save document with embedding to Supabase"""
        logger.info(f"[SupabaseService] Saving document: {metadata.get('fonte', 'unknown')}")

        if not self.client:
            logger.warning("[SupabaseService] No Supabase client, using mock doc_id")
            return f"doc_{hash(content) % 10000}"

        try:
            data = {
                "content": content,
                "embedding": embedding,
                "metadata": metadata,
                "fonte": metadata.get("fonte", "ingestion")
            }

            response = self.client.table("documents").insert(data).execute()

            if response.data:
                doc_id = response.data[0].get("id")
                logger.info(f"[SupabaseService] Document saved with ID: {doc_id}")
                return str(doc_id)
            else:
                logger.error("[SupabaseService] Save returned no data")
                return f"doc_{hash(content) % 10000}"

        except Exception as e:
            logger.error(f"[SupabaseService] Error saving document: {str(e)}")
            return f"doc_{hash(content) % 10000}"

    async def search_hybrid(self, embedding: List[float], query: str, k: int = 10, score_threshold: float = 0.0) -> List[Document]:
        """Perform hybrid (semantic + full-text) search on Supabase"""
        logger.info(f"[SupabaseService] Searching hybrid for: {query[:50]}... (k={k})")

        if not self.client:
            logger.warning("[SupabaseService] No Supabase client, returning mock results")
            mock_chunks = [
                {
                    "id": "chunk_1",
                    "content": "Como fazer backup: Acesse Configurações > Backup > Iniciar backup",
                    "embedding": embedding,
                    "score": 0.85,
                    "metadata": {"fonte": "manual"}
                },
                {
                    "id": "chunk_2",
                    "content": "O backup pode levar até 30 minutos dependendo do volume",
                    "embedding": embedding,
                    "score": 0.72,
                    "metadata": {"fonte": "faq"}
                },
                {
                    "id": "chunk_3",
                    "content": "Todos os dados são criptografados durante o backup",
                    "embedding": embedding,
                    "score": 0.65,
                    "metadata": {"fonte": "segurança"}
                },
            ]
            return [Document(**chunk) for chunk in mock_chunks]

        try:
            response = self.client.rpc(
                "match_documents_hybrid",
                {
                    "query_embedding": embedding,
                    "query_text": query,
                    "semantic_weight": 0.6,
                    "full_text_weight": 0.4,
                    "match_count": k,
                    "match_threshold": score_threshold
                }
            ).execute()

            documents = []
            if response.data:
                for item in response.data:
                    doc = Document(
                        id=item.get("id"),
                        content=item.get("content"),
                        embedding=item.get("embedding", embedding),
                        metadata=item.get("metadata", {}),
                        score=item.get("score", 0.0)
                    )
                    documents.append(doc)

            logger.info(f"[SupabaseService] Found {len(documents)} documents (score_threshold={score_threshold})")
            return documents

        except Exception as e:
            logger.error(f"[SupabaseService] Error searching hybrid: {str(e)}")
            return []

    async def get_document(self, doc_id: str) -> Optional[Document]:
        """Get a single document by ID"""
        logger.info(f"[SupabaseService] Getting document: {doc_id}")

        if not self.client:
            logger.warning("[SupabaseService] No Supabase client, returning None")
            return None

        try:
            response = self.client.table("documents").select("*").eq("id", doc_id).execute()

            if response.data:
                item = response.data[0]
                doc = Document(
                    id=item.get("id"),
                    content=item.get("content"),
                    embedding=item.get("embedding"),
                    metadata=item.get("metadata", {}),
                    score=item.get("score")
                )
                logger.info(f"[SupabaseService] Retrieved document: {doc_id}")
                return doc
            else:
                logger.warning(f"[SupabaseService] Document not found: {doc_id}")
                return None

        except Exception as e:
            logger.error(f"[SupabaseService] Error getting document: {str(e)}")
            return None
