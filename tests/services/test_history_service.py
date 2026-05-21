import pytest
from unittest.mock import AsyncMock
from src.services.history_service import HistoryService
from src.models.schemas import HistoryEntry


@pytest.fixture
def mock_supabase():
    """Provide mock Supabase client"""
    return AsyncMock()


@pytest.fixture
def service(mock_supabase):
    """Provide HistoryService with mocked client"""
    return HistoryService(mock_supabase)


@pytest.mark.asyncio
async def test_save_interaction_success(service):
    """Test saving a conversation interaction"""
    result = await service.save_interaction(
        user_id="user_123",
        pergunta="Como fazer backup?",
        resposta="Acesse Configurações > Backup",
        modo="resposta-cliente",
        score=0.85,
        chunks_used=2,
        processing_time_ms=450
    )

    assert result is not None
    assert isinstance(result, str)
    assert len(result) > 0
    assert result.startswith("history_")


@pytest.mark.asyncio
async def test_get_recent_history(service):
    """Test fetching recent conversation history"""
    history = await service.get_recent_history(user_id="user_123", limit=5)

    assert isinstance(history, list)
    assert len(history) > 0
    assert all(isinstance(h, HistoryEntry) for h in history)

    # Verify HistoryEntry fields
    first_entry = history[0]
    assert first_entry.user_id == "user_123"
    assert first_entry.pergunta is not None
    assert first_entry.resposta is not None
    assert first_entry.modo == "orientacao"
    assert first_entry.score == 0.85


@pytest.mark.asyncio
async def test_format_history_for_prompt(service):
    """Test formatting history for prompt injection"""
    formatted = await service.format_history_for_prompt(user_id="user_123", limit=5)

    assert isinstance(formatted, str)
    # Should always have expected structure when limit > 0
    assert "HISTÓRICO RECENTE" in formatted
    assert "Pergunta:" in formatted
    assert "[1]" in formatted  # Should have at least first entry formatted


@pytest.mark.asyncio
async def test_clear_old_history(service):
    """Test clearing old conversation entries"""
    result = await service.clear_old_history(days_to_keep=30)

    assert isinstance(result, int)
    assert result >= 0


@pytest.mark.asyncio
async def test_format_history_empty_on_no_history(service):
    """Test format_history handles empty history gracefully"""
    # This tests edge case: new user with no history
    formatted = await service.format_history_for_prompt(user_id="new_user_999", limit=5)

    # Should return a valid string type (can be empty or formatted)
    assert isinstance(formatted, str)


@pytest.mark.asyncio
async def test_get_recent_history_respects_limit(service):
    """Test that get_recent_history respects limit parameter"""
    history_3 = await service.get_recent_history(user_id="user_123", limit=3)
    history_5 = await service.get_recent_history(user_id="user_123", limit=5)

    # Mock returns min(limit, 3) entries, so:
    assert len(history_3) == min(3, 3)
    assert len(history_5) == min(5, 3)
