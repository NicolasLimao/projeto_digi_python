import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from src.services.openai_service import OpenAIService


@pytest.fixture
def openai_service_with_key():
    """OpenAIService with mock API key"""
    return OpenAIService(api_key="sk-test-12345")


@pytest.fixture
def openai_service_without_key():
    """OpenAIService without API key (fallback to mock)"""
    return OpenAIService(api_key="")


@pytest.mark.asyncio
async def test_classify_with_api_key(openai_service_with_key):
    """Test classify with mocked OpenAI API"""
    with patch.object(openai_service_with_key.client.chat.completions, 'create', new_callable=AsyncMock) as mock_create:
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="orientacao"))]
        mock_create.return_value = mock_response

        result = await openai_service_with_key.classify("Como fazer backup?")

        assert result == "orientacao"
        mock_create.assert_called_once()


@pytest.mark.asyncio
async def test_classify_without_api_key(openai_service_without_key):
    """Test classify falls back to mock without API key"""
    result = await openai_service_without_key.classify("Qual é o bug?")

    assert result == "bug"


@pytest.mark.asyncio
async def test_classify_resposta_cliente_keyword(openai_service_without_key):
    """Test classify returns resposta-cliente for matching query"""
    result = await openai_service_without_key.classify("Preciso de resposta para cliente")

    assert result == "resposta-cliente"


@pytest.mark.asyncio
async def test_validate_scope_within_scope(openai_service_without_key):
    """Test scope validation for Digisac query"""
    result = await openai_service_without_key.validate_scope("Como usar o Digisac?")

    assert result["dentro_do_escopo"] is True


@pytest.mark.asyncio
async def test_validate_scope_out_of_scope(openai_service_without_key):
    """Test scope validation for out-of-scope query"""
    result = await openai_service_without_key.validate_scope("Como fazer um bolo?")

    assert result["dentro_do_escopo"] is False
    assert "bolo" in result.get("motivo", "").lower()


@pytest.mark.asyncio
async def test_get_embeddings_without_api_key(openai_service_without_key):
    """Test get_embeddings returns mock embedding without API key"""
    embedding = await openai_service_without_key.get_embeddings("test text")

    assert isinstance(embedding, list)
    assert len(embedding) == 1536
    assert all(x == 0.1 for x in embedding)


@pytest.mark.asyncio
async def test_get_embeddings_returns_list(openai_service_with_key):
    """Test get_embeddings returns a list"""
    with patch.object(openai_service_with_key.client.embeddings, 'create', new_callable=AsyncMock) as mock_create:
        mock_response = MagicMock()
        mock_response.data = [MagicMock(embedding=[0.1, 0.2, 0.3] * 512)]
        mock_create.return_value = mock_response

        embedding = await openai_service_with_key.get_embeddings("test")

        assert isinstance(embedding, list)
        assert len(embedding) == 1536


@pytest.mark.asyncio
async def test_generate_response_without_chunks(openai_service_without_key):
    """Test generate_response returns mock response without API key"""
    response = await openai_service_without_key.generate_response(
        query="test",
        chunks=[],
        mode="orientacao"
    )

    assert "[MOCK]" in response


@pytest.mark.asyncio
async def test_generate_response_with_chunks(openai_service_with_key):
    """Test generate_response with chunks"""
    from src.models.schemas import Document

    chunks = [
        Document(id="1", content="Como fazer backup", embedding=[], metadata={})
    ]

    with patch.object(openai_service_with_key.client.chat.completions, 'create', new_callable=AsyncMock) as mock_create:
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Resposta de teste"))]
        mock_create.return_value = mock_response

        response = await openai_service_with_key.generate_response(
            query="Como fazer backup?",
            chunks=chunks,
            mode="orientacao"
        )

        assert response == "Resposta de teste"
        mock_create.assert_called_once()


@pytest.mark.asyncio
async def test_format_response_orientacao(openai_service_without_key):
    """Test format_response adds bullets for orientacao mode"""
    response = "Passo 1\nPasso 2\nPasso 3"

    formatted = await openai_service_without_key.format_response(response, "orientacao")

    assert formatted.startswith("- ")


@pytest.mark.asyncio
async def test_format_response_bug(openai_service_without_key):
    """Test format_response adds ERROR header for bug mode"""
    response = "Algo deu errado"

    formatted = await openai_service_without_key.format_response(response, "bug")

    assert "ERRO" in formatted or "ERROR" in formatted
