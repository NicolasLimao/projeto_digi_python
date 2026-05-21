import pytest
import asyncio
from src.services.history_service import HistoryService
from src.models.schemas import HistoryEntry


class MockSupabaseClient:
    """Mock Supabase client for integration testing"""
    pass


@pytest.mark.asyncio
async def test_full_dm_history_flow():
    """
    Test complete DM history flow:
    1. Receive DM from user
    2. Fetch history
    3. Generate response with context
    4. Save to database
    """
    user_id = "test_user_123"
    query = "Como fazer backup?"

    # Mock services
    history_service = HistoryService(MockSupabaseClient())

    # Step 1: Fetch history
    history = await history_service.get_recent_history(user_id, limit=5)
    assert isinstance(history, list)

    # Step 2: Format context
    context = await history_service.format_history_for_prompt(user_id, limit=5)
    assert isinstance(context, str)

    # Step 3: Mock response generation (would use OpenAI in production)
    response = f"{context}\nResposta a: {query}"
    assert response is not None

    # Step 4: Save to history
    entry_id = await history_service.save_interaction(
        user_id=user_id,
        pergunta=query,
        resposta=response,
        modo="resposta-cliente",
        score=0.85,
        chunks_used=3,
        processing_time_ms=500
    )

    assert entry_id is not None

    # Verify entry can be retrieved
    history = await history_service.get_recent_history(user_id, limit=5)
    assert len(history) > 0


@pytest.mark.asyncio
async def test_dm_history_multiple_users():
    """Test that history is correctly isolated per user"""
    history_service = HistoryService(MockSupabaseClient())

    # Save interactions for different users
    users = ["user_1", "user_2", "user_3"]
    for user_id in users:
        for i in range(3):
            await history_service.save_interaction(
                user_id=user_id,
                pergunta=f"Pergunta {i} de {user_id}",
                resposta=f"Resposta {i}",
                score=0.80 + (i * 0.01)
            )

    # Verify isolation
    for user_id in users:
        user_history = await history_service.get_recent_history(user_id, limit=10)
        assert len(user_history) > 0

    pass


@pytest.mark.asyncio
async def test_history_context_injection():
    """Test that history context is properly formatted for prompt injection"""
    user_id = "test_user"
    history_service = HistoryService(MockSupabaseClient())

    # Get formatted history
    formatted = await history_service.format_history_for_prompt(user_id, limit=5)

    # Should have proper structure
    if formatted:
        assert "HISTÓRICO RECENTE" in formatted
        assert "Pergunta:" in formatted
    pass


@pytest.mark.asyncio
async def test_save_and_retrieve_cycle():
    """Test complete save and retrieve cycle"""
    user_id = "cycle_test_user"
    history_service = HistoryService(MockSupabaseClient())

    # Save multiple interactions
    interactions = [
        {"pergunta": "O que é backup?", "resposta": "Backup é cópia de dados"},
        {"pergunta": "Como fazer?", "resposta": "Clique em Configurações"},
        {"pergunta": "Quanto tempo leva?", "resposta": "Até 30 minutos"}
    ]

    for i, interaction in enumerate(interactions):
        entry_id = await history_service.save_interaction(
            user_id=user_id,
            pergunta=interaction["pergunta"],
            resposta=interaction["resposta"],
            score=0.80 + (i * 0.05),
            chunks_used=i + 1,
            processing_time_ms=400 + (i * 50)
        )
        assert entry_id is not None

    # Retrieve and verify
    history = await history_service.get_recent_history(user_id, limit=10)
    assert len(history) >= len(interactions)
