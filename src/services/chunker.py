from typing import List
from src.logger import get_logger

logger = get_logger(__name__)


class ChunkerService:
    MAX_CHUNK_SIZE = 800
    MIN_CHUNK_SIZE = 30

    @staticmethod
    def chunk_semantic(text: str) -> List[str]:
        """Chunking semântico por heading > parágrafo > tamanho fixo"""
        logger.info(f"Chunking semantic text ({len(text)} chars)")

        # Tentar dividir por heading
        sections = text.split(/(?=^#{1,3}\s)/m)
        sections = [s.strip() for s in sections if len(s.strip()) > ChunkerService.MIN_CHUNK_SIZE]

        if len(sections) > 1:
            logger.info(f"Found {len(sections)} sections by heading")
            return sections

        # Fallback: dividir por parágrafo
        sections = text.split("\n\n")
        sections = [s.strip() for s in sections if len(s.strip()) > ChunkerService.MIN_CHUNK_SIZE]

        if len(sections) > 1:
            logger.info(f"Found {len(sections)} sections by paragraph")
            return sections

        # Fallback: dividir por tamanho fixo
        words = text.split()
        chunks = []
        i = 0

        while i < len(words):
            chunk = " ".join(words[i:i + ChunkerService.MAX_CHUNK_SIZE])
            if len(chunk) > ChunkerService.MIN_CHUNK_SIZE:
                chunks.append(chunk)
            i += ChunkerService.MAX_CHUNK_SIZE - 100  # overlap

        logger.info(f"Created {len(chunks)} chunks by fixed size")
        return chunks if chunks else [text]
