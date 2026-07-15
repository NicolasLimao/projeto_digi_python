import re

from src.logger import get_logger

logger = get_logger(__name__)
HEADING_RE = re.compile(r"(?=^#{1,3}\s)", re.MULTILINE)
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


def _split_oversized(text: str, max_chars: int, overlap_chars: int) -> list[str]:
    sentences = [sentence.strip() for sentence in SENTENCE_RE.split(text) if sentence.strip()]
    chunks: list[str] = []
    current = ""

    for sentence in sentences:
        if len(sentence) > max_chars:
            if current:
                chunks.append(current)
                current = ""
            step = max(1, max_chars - overlap_chars)
            chunks.extend(
                sentence[start : start + max_chars].strip()
                for start in range(0, len(sentence), step)
                if sentence[start : start + max_chars].strip()
            )
            continue

        candidate = f"{current} {sentence}".strip()
        if current and len(candidate) > max_chars:
            chunks.append(current)
            overlap = current[-overlap_chars:].lstrip() if overlap_chars else ""
            current = f"{overlap} {sentence}".strip()
        else:
            current = candidate

    if current:
        chunks.append(current)
    return chunks


def chunk_text(
    text: str,
    *,
    max_chars: int = 1_000,
    min_chars: int = 30,
    overlap_chars: int = 120,
) -> list[str]:
    """Split text by headings and paragraphs while bounding every chunk.

    The previous implementation accidentally contained JavaScript regex syntax,
    which made this entire module impossible to import.
    """

    normalized = text.replace("\r\n", "\n").strip()
    if not normalized:
        return []
    if max_chars <= 0 or min_chars < 0 or overlap_chars < 0:
        raise ValueError("Chunk sizes must be non-negative and max_chars must be positive")
    if overlap_chars >= max_chars:
        raise ValueError("overlap_chars must be smaller than max_chars")

    sections = [section.strip() for section in HEADING_RE.split(normalized) if section.strip()]
    units: list[str] = []
    for section in sections:
        units.extend(part.strip() for part in re.split(r"\n\s*\n", section) if part.strip())

    chunks: list[str] = []
    current: list[str] = []
    current_size = 0
    for unit in units:
        if len(unit) > max_chars:
            if current:
                chunks.append("\n\n".join(current))
                current, current_size = [], 0
            chunks.extend(_split_oversized(unit, max_chars, overlap_chars))
            continue

        additional = len(unit) + (2 if current else 0)
        if current and current_size + additional > max_chars:
            chunks.append("\n\n".join(current))
            current, current_size = [unit], len(unit)
        else:
            current.append(unit)
            current_size += additional

    if current:
        chunks.append("\n\n".join(current))

    result = [chunk for chunk in chunks if len(chunk) >= min_chars]
    if not result and len(normalized) < min_chars:
        result = [normalized]
    logger.info("Chunking completed", extra={"extras": {"chars": len(text), "chunks": len(result)}})
    return result


class ChunkerService:
    MAX_CHUNK_SIZE = 1_000
    MIN_CHUNK_SIZE = 30

    @staticmethod
    def chunk_semantic(text: str) -> list[str]:
        return chunk_text(
            text,
            max_chars=ChunkerService.MAX_CHUNK_SIZE,
            min_chars=ChunkerService.MIN_CHUNK_SIZE,
        )
