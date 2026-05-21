import pytest
from unittest.mock import AsyncMock, MagicMock
from src.agents.rag_agent import RAGAgent
from src.services.openai_service import OpenAIService
from src.services.supabase_service import SupabaseService
from src.models.schemas import Document


@pytest.fixture
def mock_openai_service():
    """Mock OpenAI service"""
    service = MagicMock(spec=OpenAIService)
    service.get_embeddings = AsyncMock(return_value=[0.1] * 1536)
    service.generate_response = AsyncMock(return_value="Test response")
    service.format_response = AsyncMock(return_value="Formatted response")
    return service


@pytest.fixture
def mock_supabase_service():
    """Mock Supabase service"""
    service = MagicMock(spec=SupabaseService)

    doc1 = Document(
        id="doc_1",
        content="Backup content",
        embedding=[0.1] * 1536,
        metadata={"fonte": "manual"},
        score=0.9
    )
    doc2 = Document(
        id="doc_2",
        content="Restore content",
        embedding=[0.1] * 1536,
        metadata={"fonte": "faq"},
        score=0.7
    )

    service.search_hybrid = AsyncMock(return_value=[doc1, doc2])
    return service


@pytest.fixture
def rag_agent(mock_openai_service, mock_supabase_service):
    """RAGAgent with mocked services"""
    return RAGAgent(mock_openai_service, mock_supabase_service)


@pytest.mark.asyncio
async def test_execute_returns_dict(rag_agent):
    """Test RAGAgent.execute returns a dictionary"""
    result = await rag_agent.execute(query="How to backup?")

    assert isinstance(result, dict)
    assert "response" in result
    assert "score" in result
    assert "chunks_used" in result
    assert "mode" in result


@pytest.mark.asyncio
async def test_execute_calls_get_embeddings(rag_agent, mock_openai_service):
    """Test execute calls get_embeddings"""
    await rag_agent.execute(query="test")

    mock_openai_service.get_embeddings.assert_called_once()


@pytest.mark.asyncio
async def test_execute_calls_search_hybrid(rag_agent, mock_supabase_service):
    """Test execute calls search_hybrid with embeddings"""
    await rag_agent.execute(query="test")

    mock_supabase_service.search_hybrid.assert_called_once()
    call_args = mock_supabase_service.search_hybrid.call_args
    assert "embedding" in call_args.kwargs


@pytest.mark.asyncio
async def test_execute_respects_k_limit(rag_agent, mock_supabase_service):
    """Test execute passes k parameter to search_hybrid"""
    await rag_agent.execute(query="test", k=3)

    call_args = mock_supabase_service.search_hybrid.call_args
    assert call_args.kwargs.get("k") == 3


@pytest.mark.asyncio
async def test_execute_calculates_average_score(rag_agent):
    """Test execute calculates average similarity score"""
    result = await rag_agent.execute(query="test")

    assert result["score"] == pytest.approx((0.9 + 0.7) / 2, 0.01)


@pytest.mark.asyncio
async def test_execute_counts_chunks(rag_agent):
    """Test execute counts retrieved chunks"""
    result = await rag_agent.execute(query="test")

    assert result["chunks_used"] == 2


@pytest.mark.asyncio
async def test_execute_with_no_results(rag_agent, mock_supabase_service):
    """Test execute handles no search results"""
    mock_supabase_service.search_hybrid.return_value = []

    result = await rag_agent.execute(query="unknown topic")

    assert result["score"] == 0.0
    assert result["chunks_used"] == 0
    # Response is formatted through formatter agent
    assert isinstance(result["response"], str)


@pytest.mark.asyncio
async def test_execute_handles_exception(rag_agent, mock_openai_service):
    """Test execute handles exceptions gracefully"""
    mock_openai_service.get_embeddings.side_effect = Exception("API Error")

    result = await rag_agent.execute(query="test")

    assert "Erro" in result["response"]
    assert result["score"] == 0.0
    assert result["chunks_used"] == 0


@pytest.mark.asyncio
async def test_execute_includes_mode(rag_agent):
    """Test execute result includes requested mode"""
    result = await rag_agent.execute(query="test", mode="resposta-cliente")

    assert result["mode"] == "resposta-cliente"


@pytest.mark.asyncio
async def test_execute_calls_generate_response_with_chunks(rag_agent, mock_openai_service):
    """Test generate_response receives retrieved documents"""
    await rag_agent.execute(query="test")

    mock_openai_service.generate_response.assert_called_once()
    call_args = mock_openai_service.generate_response.call_args
    assert len(call_args[0][1]) == 2  # Two documents passed


@pytest.mark.asyncio
async def test_execute_uses_rag_mode(rag_agent):
    """Test execute uses correct mode"""
    result = await rag_agent.execute(query="test", mode="resposta-cliente")

    assert result["mode"] == "resposta-cliente"
