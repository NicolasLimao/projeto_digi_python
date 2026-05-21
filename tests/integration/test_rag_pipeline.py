import pytest
from unittest.mock import AsyncMock, MagicMock
from src.pipeline.rag_pipeline import RAGPipeline
from src.agents.classifier import ClassifierAgent
from src.agents.scope_validator import ScopeValidatorAgent
from src.agents.rag_agent import RAGAgent
from src.agents.formatter_agent import FormatterAgent
from src.services.history_service import HistoryService
from src.models.schemas import QueryResponse


@pytest.fixture
def mock_classifier():
    """Mock classifier agent"""
    agent = MagicMock(spec=ClassifierAgent)
    agent.execute = AsyncMock(return_value="orientacao")
    return agent


@pytest.fixture
def mock_validator():
    """Mock scope validator agent"""
    agent = MagicMock(spec=ScopeValidatorAgent)
    agent.execute = AsyncMock(return_value={"dentro_do_escopo": True})
    return agent


@pytest.fixture
def mock_rag_agent():
    """Mock RAG agent"""
    agent = MagicMock(spec=RAGAgent)
    agent.execute = AsyncMock(return_value={
        "response": "Test response",
        "score": 0.85,
        "chunks_used": 2,
        "documents": []
    })
    return agent


@pytest.fixture
def mock_formatter():
    """Mock formatter agent"""
    agent = MagicMock(spec=FormatterAgent)
    agent.execute = AsyncMock(return_value="Formatted response")
    return agent


@pytest.fixture
def mock_history_service():
    """Mock history service"""
    service = MagicMock(spec=HistoryService)
    service.save_interaction = AsyncMock(return_value="history_123")
    return service


@pytest.fixture
def pipeline(mock_classifier, mock_validator, mock_rag_agent, mock_formatter, mock_history_service):
    """RAGPipeline with all mocked dependencies"""
    return RAGPipeline(mock_classifier, mock_validator, mock_rag_agent, mock_formatter, mock_history_service)


@pytest.mark.asyncio
async def test_process_returns_query_response(pipeline):
    """Test process returns QueryResponse object"""
    result = await pipeline.process(query="test query", user_id="user_123")

    assert isinstance(result, QueryResponse)
    assert hasattr(result, "response")
    assert hasattr(result, "mode")
    assert hasattr(result, "score")
    assert hasattr(result, "chunks_used")
    assert hasattr(result, "processing_time_ms")


@pytest.mark.asyncio
async def test_process_calls_classifier(pipeline, mock_classifier):
    """Test process calls classifier agent"""
    await pipeline.process(query="test", user_id="user_123")

    mock_classifier.execute.assert_called_once()


@pytest.mark.asyncio
async def test_process_calls_validator(pipeline, mock_validator):
    """Test process calls scope validator"""
    await pipeline.process(query="test", user_id="user_123")

    mock_validator.execute.assert_called_once()


@pytest.mark.asyncio
async def test_process_calls_rag_if_in_scope(pipeline, mock_rag_agent):
    """Test process calls RAG agent when in scope"""
    await pipeline.process(query="test", user_id="user_123")

    mock_rag_agent.execute.assert_called_once()


@pytest.mark.asyncio
async def test_process_skips_rag_if_out_of_scope(pipeline, mock_validator, mock_rag_agent):
    """Test process skips RAG when out of scope"""
    mock_validator.execute.return_value = {
        "dentro_do_escopo": False,
        "motivo": "Out of scope"
    }

    await pipeline.process(query="test", user_id="user_123")

    mock_rag_agent.execute.assert_not_called()


@pytest.mark.asyncio
async def test_process_calls_formatter(pipeline, mock_formatter):
    """Test process calls formatter agent"""
    await pipeline.process(query="test", user_id="user_123")

    mock_formatter.execute.assert_called_once()


@pytest.mark.asyncio
async def test_process_saves_to_history(pipeline, mock_history_service):
    """Test process saves interaction to history"""
    await pipeline.process(query="test query", user_id="user_123")

    mock_history_service.save_interaction.assert_called_once()
    call_args = mock_history_service.save_interaction.call_args
    assert call_args.kwargs.get("user_id") == "user_123"
    assert call_args.kwargs.get("pergunta") == "test query"


@pytest.mark.asyncio
async def test_process_with_explicit_mode(pipeline, mock_classifier):
    """Test process respects explicit mode parameter"""
    await pipeline.process(query="test", user_id="user_123", mode="resposta-cliente")

    mock_classifier.execute.assert_not_called()


@pytest.mark.asyncio
async def test_process_out_of_scope_returns_early(pipeline, mock_validator, mock_formatter):
    """Test out-of-scope query doesn't call formatter"""
    mock_validator.execute.return_value = {
        "dentro_do_escopo": False,
        "motivo": "Test"
    }

    result = await pipeline.process(query="test", user_id="user_123")

    assert "fora do escopo" in result.response.lower()


@pytest.mark.asyncio
async def test_process_handles_exception(pipeline, mock_rag_agent, mock_history_service):
    """Test process handles exceptions gracefully"""
    mock_rag_agent.execute.side_effect = Exception("Test error")

    result = await pipeline.process(query="test", user_id="user_123")

    assert "Erro" in result.response
    assert result.score == 0.0
    mock_history_service.save_interaction.assert_called_once()


@pytest.mark.asyncio
async def test_process_measures_processing_time(pipeline):
    """Test process includes processing time"""
    result = await pipeline.process(query="test", user_id="user_123")

    assert result.processing_time_ms >= 0
    assert isinstance(result.processing_time_ms, (int, float))


@pytest.mark.asyncio
async def test_process_includes_response_text(pipeline):
    """Test process result includes formatted response"""
    result = await pipeline.process(query="test", user_id="user_123")

    assert result.response == "Formatted response"


@pytest.mark.asyncio
async def test_process_uses_rag_score(pipeline, mock_rag_agent):
    """Test process includes RAG agent score"""
    mock_rag_agent.execute.return_value = {
        "response": "Response",
        "score": 0.92,
        "chunks_used": 3,
        "documents": []
    }

    result = await pipeline.process(query="test", user_id="user_123")

    assert result.score == 0.92


@pytest.mark.asyncio
async def test_process_counts_chunks(pipeline, mock_rag_agent):
    """Test process includes chunk count"""
    mock_rag_agent.execute.return_value = {
        "response": "Response",
        "score": 0.85,
        "chunks_used": 5,
        "documents": []
    }

    result = await pipeline.process(query="test", user_id="user_123")

    assert result.chunks_used == 5
