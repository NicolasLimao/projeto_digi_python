import pytest

from src.services.chunker import ChunkerService, chunk_text


def test_empty_text_returns_no_chunks():
    assert chunk_text("   ") == []


def test_headings_and_paragraphs_are_preserved():
    text = (
        "# Instalação\n\n"
        + "Primeiro parágrafo. " * 10
        + "\n\n## Uso\n\n"
        + "Segundo parágrafo. " * 10
    )
    chunks = chunk_text(text, max_chars=300)
    assert len(chunks) >= 2
    assert any("Instalação" in chunk for chunk in chunks)
    assert any("Uso" in chunk for chunk in chunks)


def test_every_chunk_respects_maximum_size():
    chunks = chunk_text("Sentença longa. " * 300, max_chars=180, overlap_chars=20)
    assert chunks
    assert all(len(chunk) <= 180 for chunk in chunks)


def test_tiny_text_is_not_lost():
    assert chunk_text("curto", min_chars=30) == ["curto"]


def test_invalid_overlap_is_rejected():
    with pytest.raises(ValueError, match="overlap"):
        chunk_text("texto", max_chars=100, overlap_chars=100)


def test_service_wrapper_uses_valid_python_implementation():
    assert ChunkerService.chunk_semantic("Um texto simples e suficientemente longo para um chunk.")
