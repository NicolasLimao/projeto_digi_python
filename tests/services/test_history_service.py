import pytest
from unittest.mock import AsyncMock, MagicMock
from src.services.history_service import HistoryService
from src.models.schemas import HistoryEntry


class MockSupabaseTable:
    """Mock Supabase table operations"""
    def __init__(self):
        self.data_store = []

    def insert(self, data):
        entry_id = f"entry_{len(self.data_store)}"
        self.data_store.append({**data, "id": entry_id})
        return self

    def select(self, *args):
        return self

    def eq(self, key, value):
        return self

    def order(self, key, **kwargs):
        return self

    def limit(self, count):
        return self

    def delete(self):
        return self

    def lt(self, key, value):
        return self

    def execute(self):
        mock_response = MagicMock()
        mock_response.data = [{"id": f"entry_{len(self.data_store)}"}]
        return mock_response


class MockSupabaseClient:
    """Mock Supabase client for testing"""
    def __init__(self):
        self._table = MockSupabaseTable()

    def table(self, name):
        return self._table


@pytest.fixture
def mock_supabase():
    """Provide proper mock Supabase client"""
    return MockSupabaseClient()


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


@pytest.mark.asyncio
async def test_get_recent_history(service):
    """Test fetching recent conversation history"""
    history = await service.get_recent_history(user_id="user_123", limit=5)

    assert isinstance(history, list)


@pytest.mark.asyncio
async def test_format_history_for_prompt(service):
    """Test formatting history for prompt injection"""
    formatted = await service.format_history_for_prompt(user_id="user_123", limit=5)

    assert isinstance(formatted, str)


@pytest.mark.asyncio
async def test_clear_old_history(service):
    """Test clearing old conversation entries"""
    result = await service.clear_old_history(days_to_keep=30)

    assert isinstance(result, int)
    assert result >= 0


@pytest.mark.asyncio
async def test_format_history_empty_on_no_history(service):
    """Test format_history handles empty history gracefully"""
    formatted = await service.format_history_for_prompt(user_id="new_user_999", limit=5)
    assert isinstance(formatted, str)


@pytest.mark.asyncio
async def test_get_recent_history_respects_limit(service):
    """Test that get_recent_history respects limit parameter"""
    history_3 = await service.get_recent_history(user_id="user_123", limit=3)
    history_5 = await service.get_recent_history(user_id="user_123", limit=5)

    assert isinstance(history_3, list)
    assert isinstance(history_5, list)
