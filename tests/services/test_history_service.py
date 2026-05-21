import pytest
from src.services.history_service import HistoryService
from src.models.schemas import HistoryEntry


class MockSupabaseClient:
    """Mock Supabase client for testing"""
    pass


@pytest.mark.asyncio
async def test_save_interaction_success():
    """Test saving a conversation interaction"""
    service = HistoryService(MockSupabaseClient())

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


@pytest.mark.asyncio
async def test_get_recent_history():
    """Test fetching recent conversation history"""
    service = HistoryService(MockSupabaseClient())

    history = await service.get_recent_history(user_id="user_123", limit=5)

    assert isinstance(history, list)
    assert all(isinstance(h, HistoryEntry) for h in history)


@pytest.mark.asyncio
async def test_format_history_for_prompt():
    """Test formatting history for prompt injection"""
    service = HistoryService(MockSupabaseClient())

    formatted = await service.format_history_for_prompt(user_id="user_123", limit=5)

    assert isinstance(formatted, str)
    if formatted:
        assert "HISTÓRICO RECENTE" in formatted
        assert "Pergunta:" in formatted


@pytest.mark.asyncio
async def test_clear_old_history():
    """Test clearing old conversation entries"""
    service = HistoryService(MockSupabaseClient())

    result = await service.clear_old_history(days_to_keep=30)

    assert isinstance(result, int)
    assert result >= 0
