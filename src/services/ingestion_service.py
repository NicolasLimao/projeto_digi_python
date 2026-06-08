"""
IngestionService: substitui o workflow de ingestão do n8n.
Fluxo: recebe texto + anexos -> extrai (PyMuPDF / Mistral OCR fallback) -> chunka ->
embedding em batch (OpenAI) -> insert em batch no Supabase via service_role.
"""
import re
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any
import httpx
from supabase import create_client
from openai import AsyncOpenAI
from src.config import settings
from src.logger import get_logger

logger = get_logger(__name__)

PDF_MIMES = {"application/pdf"}
TEXT_MIMES = {"text/plain", "text/markdown"}

CHUNK_TARGET_SIZE = 1000     # chars por chunk (alvo)
EMBED_BATCH = 100            # chunks por chamada de embedding
INSERT_BATCH = 100           # rows por insert no Supabase


# ---------------------------- Chunking ---------------------------- #

def chunk_text(text: str, target_size: int = CHUNK_TARGET_SIZE) -> List[str]:
    """Chunking por parágrafos, agregando até target_size chars.
    Parágrafos muito longos são quebrados em sentenças."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: List[str] = []
    current: List[str] = []
    current_size = 0

    for p in paragraphs:
        if len(p) > target_size * 1.5:
            if current:
                chunks.append("\n\n".join(current))
                current, current_size = [], 0
            chunks.extend(_split_long_paragraph(p, target_size))
            continue
        if current_size + len(p) > target_size and current:
            chunks.append("\n\n".join(current))
            current, current_size = [p], len(p)
        else:
            current.append(p)
            current_size += len(p) + 2

    if current:
        chunks.append("\n\n".join(current))
    return chunks


def _split_long_paragraph(p: str, target_size: int) -> List[str]:
    sentences = re.split(r"(?<=[.!?])\s+", p)
    out: List[str] = []
    cur = ""
    for s in sentences:
        if len(cur) + len(s) > target_size and cur:
            out.append(cur.strip())
            cur = s
        else:
            cur = (cur + " " + s) if cur else s
    if cur:
        out.append(cur.strip())
    return out or [p[:target_size]]


# ---------------------------- Service ---------------------------- #

class IngestionService:
    def __init__(self):
        self.openai = AsyncOpenAI(api_key=settings.openai_api_key) if settings.openai_api_key else None
        sb_key = settings.supabase_service_role_key or settings.supabase_anon_key
        self.supabase = create_client(settings.supabase_url, sb_key)
        self.using_service_role = bool(settings.supabase_service_role_key)
        self.mistral_key = settings.mistral_api_key
        logger.info(f"[IngestionService] inicializado (service_role={self.using_service_role}, mistral={bool(self.mistral_key)})")

    async def ingest(
        self,
        content: Optional[str] = None,
        attachments: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """Recebe texto + anexos, extrai/chunka/embeda/insere. Retorna totais."""
        sources: List[Dict[str, Any]] = []
        errors: List[str] = []
        total_chars = 0

        textos: List[tuple[str, str]] = []  # (texto, fonte)

        # 1) Texto inline (mensagem do Discord)
        if content and content.strip():
            textos.append((content.strip(), "discord-text"))

        # 2) Anexos
        for att in (attachments or []):
            url = att.get("url")
            filename = att.get("filename") or "anexo"
            ctype = (att.get("contentType") or "").lower()
            if not url:
                continue
            try:
                texto = await self._extract_attachment(url, ctype, filename)
                if texto:
                    textos.append((texto, filename))
                    logger.info(f"[Ingestion] Extraído {len(texto)} chars de {filename}")
                else:
                    errors.append(f"{filename}: extração vazia")
                    logger.warning(f"[Ingestion] Extração vazia em {filename}")
            except Exception as e:
                errors.append(f"{filename}: {str(e)[:120]}")
                logger.error(f"[Ingestion] Erro em {filename}: {e}")

        if not textos:
            return {"chunks_created": 0, "total_chars": 0, "sources": [], "errors": errors or ["Nada para ingerir"]}

        # 3) Chunkar
        all_chunks: List[str] = []
        for texto, src in textos:
            chs = chunk_text(texto)
            all_chunks.extend(chs)
            total_chars += len(texto)
            sources.append({"source": src, "chars": len(texto), "chunks": len(chs)})

        if not all_chunks:
            return {"chunks_created": 0, "total_chars": total_chars, "sources": sources, "errors": errors + ["Nenhum chunk gerado"]}

        # 4) Embeddings em batch
        logger.info(f"[Ingestion] Gerando embeddings para {len(all_chunks)} chunks...")
        embeddings = await self._embed_batch(all_chunks)

        # 5) Insert em batch
        timestamp = datetime.now(timezone.utc).isoformat()
        rows = [
            {
                "content": chunk,
                "embedding": emb,
                "metadata": {
                    "data": timestamp,
                    "fonte": "discord-upload",
                    "tipo_chunk": "semantico",
                    "chunk_index": idx
                }
            }
            for idx, (chunk, emb) in enumerate(zip(all_chunks, embeddings))
        ]

        inserted = 0
        for i in range(0, len(rows), INSERT_BATCH):
            batch = rows[i:i + INSERT_BATCH]
            try:
                r = self.supabase.table("documents").insert(batch).execute()
                inserted += len(r.data or [])
            except Exception as e:
                logger.error(f"[Ingestion] Erro no insert batch {i // INSERT_BATCH}: {e}")
                errors.append(f"insert batch {i // INSERT_BATCH}: {str(e)[:120]}")

        logger.info(f"[Ingestion] Inseridos {inserted}/{len(rows)} chunks ({total_chars} chars)")
        return {
            "chunks_created": inserted,
            "total_chars": total_chars,
            "sources": sources,
            "errors": errors
        }

    # ---------- Extração ---------- #

    async def _extract_attachment(self, url: str, ctype: str, filename: str) -> str:
        async with httpx.AsyncClient(timeout=60) as http:
            resp = await http.get(url)
            resp.raise_for_status()
            data = resp.content

        is_pdf = ctype in PDF_MIMES or filename.lower().endswith(".pdf")
        is_text = ctype in TEXT_MIMES or filename.lower().endswith((".txt", ".md"))

        if is_pdf:
            texto = self._extract_pdf_pymupdf(data)
            if texto.strip():
                return texto
            if self.mistral_key:
                logger.info(f"[Ingestion] PyMuPDF vazio em {filename}, tentando Mistral OCR...")
                return await self._extract_pdf_mistral(url)
            logger.warning(f"[Ingestion] PDF imagem sem MISTRAL_API_KEY: {filename}")
            return ""

        if is_text:
            return data.decode("utf-8", errors="replace")

        logger.warning(f"[Ingestion] Tipo não suportado: {ctype} ({filename})")
        return ""

    def _extract_pdf_pymupdf(self, pdf_bytes: bytes) -> str:
        try:
            import fitz  # PyMuPDF
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            partes = [page.get_text() for page in doc]
            doc.close()
            return "\n\n".join(partes)
        except Exception as e:
            logger.error(f"[Ingestion] Erro no PyMuPDF: {e}")
            return ""

    async def _extract_pdf_mistral(self, url: str) -> str:
        try:
            from mistralai import Mistral
            client = Mistral(api_key=self.mistral_key)
            resp = client.ocr.process(
                model="mistral-ocr-latest",
                document={"type": "document_url", "document_url": url}
            )
            partes = []
            for page in resp.pages:
                md = getattr(page, "markdown", None)
                if md:
                    partes.append(md)
            return "\n\n".join(partes)
        except Exception as e:
            logger.error(f"[Ingestion] Erro no Mistral OCR: {e}")
            return ""

    # ---------- Embeddings ---------- #

    async def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        if not self.openai:
            return [[0.1] * 1536 for _ in texts]
        out: List[List[float]] = []
        for i in range(0, len(texts), EMBED_BATCH):
            batch = texts[i:i + EMBED_BATCH]
            resp = await self.openai.embeddings.create(
                model="text-embedding-3-small",
                input=batch
            )
            out.extend([d.embedding for d in resp.data])
        return out
