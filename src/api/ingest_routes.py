"""Endpoint de ingestão: substitui o webhook do n8n.
POST /api/ingest com {content?, attachments?: [{url, filename, contentType}]}
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from src.services.ingestion_service import IngestionService
from src.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/api", tags=["ingest"])


class Attachment(BaseModel):
    url: str
    filename: Optional[str] = None
    contentType: Optional[str] = None


class IngestPayload(BaseModel):
    content: Optional[str] = None
    attachments: List[Attachment] = []


class IngestResult(BaseModel):
    chunks_created: int
    total_chars: int
    sources: List[Dict[str, Any]] = []
    errors: List[str] = []


@router.post("/ingest", response_model=IngestResult)
async def ingest(payload: IngestPayload):
    """Recebe texto + anexos do bot. Extrai, chunka, embeda e insere no Supabase."""
    logger.info(f"[API] Ingest: content={bool(payload.content)} attachments={len(payload.attachments)}")

    if not payload.content and not payload.attachments:
        raise HTTPException(status_code=400, detail="content ou attachments é obrigatório")

    try:
        svc = IngestionService()
        result = await svc.ingest(
            content=payload.content,
            attachments=[a.model_dump() for a in payload.attachments]
        )
        return IngestResult(**result)
    except Exception as e:
        logger.error(f"[API] Erro na ingestão: {e}")
        raise HTTPException(status_code=500, detail=f"Erro na ingestão: {str(e)}")
