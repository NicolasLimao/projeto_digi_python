from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from src.config import Settings
from src.services.ingestion_service import IngestionService, IngestionValidationError
from tests.fakes import FakeSupabaseClient


def make_http_client(content: bytes, content_type: str = "text/plain") -> httpx.AsyncClient:
    transport = httpx.MockTransport(
        lambda request: httpx.Response(
            200,
            content=content,
            headers={"content-type": content_type, "content-length": str(len(content))},
        )
    )
    return httpx.AsyncClient(transport=transport)


def make_service(**settings_overrides) -> IngestionService:
    config = Settings(environment="test", **settings_overrides)
    openai = MagicMock()
    openai.embeddings.create = AsyncMock(
        side_effect=lambda **kwargs: SimpleNamespace(
            data=[SimpleNamespace(embedding=[0.1, 0.2]) for _ in kwargs["input"]]
        )
    )
    return IngestionService(
        config,
        openai_client=openai,
        supabase_client=FakeSupabaseClient(),
        http_client=make_http_client(b"conteudo"),
    )


@pytest.mark.parametrize(
    "url",
    [
        "http://cdn.discordapp.com/file.txt",
        "https://evil.example/file.txt",
        "https://user:pass@cdn.discordapp.com/file.txt",
        "https://cdn.discordapp.com:444/file.txt",
    ],
)
def test_attachment_url_rejects_ssrf_vectors(url: str):
    with pytest.raises(IngestionValidationError):
        make_service()._validate_attachment_url(url)


def test_attachment_url_accepts_exact_allowlisted_https_host():
    url = "https://cdn.discordapp.com/attachments/file.txt"
    assert make_service()._validate_attachment_url(url) == url


@pytest.mark.asyncio
async def test_download_rejects_declared_oversize():
    client = make_http_client(b"0123456789")
    service = IngestionService(
        Settings(environment="test", ingest_max_file_bytes=5_000),
        openai_client=MagicMock(),
        supabase_client=FakeSupabaseClient(),
        http_client=client,
    )
    service.config.ingest_max_file_bytes = 5
    with pytest.raises(IngestionValidationError, match="too large"):
        await service._download_attachment("https://cdn.discordapp.com/file.txt")
    await client.aclose()


@pytest.mark.asyncio
async def test_download_rejects_redirects():
    transport = httpx.MockTransport(
        lambda request: httpx.Response(
            302,
            headers={"location": "https://evil.example/file"},
        )
    )
    client = httpx.AsyncClient(transport=transport, follow_redirects=False)
    service = IngestionService(
        Settings(environment="test"),
        openai_client=MagicMock(),
        supabase_client=FakeSupabaseClient(),
        http_client=client,
    )
    with pytest.raises(IngestionValidationError, match="redirects"):
        await service._download_attachment("https://cdn.discordapp.com/file.txt")
    await client.aclose()


@pytest.mark.asyncio
async def test_text_attachment_is_extracted():
    client = make_http_client("Olá".encode(), "text/plain")
    service = IngestionService(
        Settings(environment="test"),
        openai_client=MagicMock(),
        supabase_client=FakeSupabaseClient(),
        http_client=client,
    )
    assert (
        await service._extract_attachment(
            "https://cdn.discordapp.com/file.txt", "text/plain", "file.txt"
        )
        == "Olá"
    )
    await client.aclose()


@pytest.mark.asyncio
async def test_inline_ingestion_embeds_and_persists_chunks():
    service = make_service()
    result = await service.ingest(content="Documentação válida. " * 100)
    assert result["chunks_created"] > 0
    assert result["total_chars"] > 0
    assert result["errors"] == []


@pytest.mark.asyncio
async def test_ingestion_requires_real_service_configuration():
    service = IngestionService(
        Settings(environment="test"),
        http_client=make_http_client(b"texto"),
    )
    with pytest.raises(RuntimeError, match="must be configured"):
        await service.ingest(content="texto")
    await service.http.aclose()


@pytest.mark.asyncio
async def test_total_extracted_text_is_bounded():
    service = make_service(ingest_max_total_chars=1_000)
    with pytest.raises(IngestionValidationError, match="exceeds"):
        await service.ingest(content="x" * 1_001)
