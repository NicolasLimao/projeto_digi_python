import pytest
from unittest.mock import AsyncMock, MagicMock
from src.services.supabase_service import SupabaseService
from src.models.schemas import Document


@pytest.fixture
def supabase_service_with_credentials():
    """SupabaseService with test credentials"""
    return SupabaseService(url="https://test.supabase.co", key="test-key")


@pytest.fixture
def supabase_service_without_credentials():
    """SupabaseService without credentials (fallback to mock)"""
    return SupabaseService(url="", key="")


@pytest.mark.asyncio
async def test_save_document_without_credentials(supabase_service_without_credentials):
    """Test save_document returns mock ID without Supabase"""
    doc_id = await supabase_service_without_credentials.save_document(
        content="Test content",
        embedding=[0.1] * 1536,
        metadata={"fonte": "test"}
    )

    assert doc_id.startswith("doc_")
    assert isinstance(doc_id, str)


@pytest.mark.asyncio
async def test_search_hybrid_without_credentials(supabase_service_without_credentials):
    """Test search_hybrid returns mock documents without Supabase"""
    documents = await supabase_service_without_credentials.search_hybrid(
        embedding=[0.1] * 1536,
        query="backup",
        k=5
    )

    assert isinstance(documents, list)
    assert len(documents) == 3
    assert all(isinstance(doc, Document) for doc in documents)
    assert all(hasattr(doc, "content") for doc in documents)


@pytest.mark.asyncio
async def test_search_hybrid_respects_k_limit(supabase_service_without_credentials):
    """Test search_hybrid returns at most k documents"""
    documents = await supabase_service_without_credentials.search_hybrid(
        embedding=[0.1] * 1536,
        query="test",
        k=2
    )

    assert len(documents) <= 2


@pytest.mark.asyncio
async def test_search_hybrid_includes_score(supabase_service_without_credentials):
    """Test search_hybrid documents include similarity score"""
    documents = await supabase_service_without_credentials.search_hybrid(
        embedding=[0.1] * 1536,
        query="test",
        k=5
    )

    assert all(doc.score is not None for doc in documents)
    assert all(0.0 <= doc.score <= 1.0 for doc in documents)


@pytest.mark.asyncio
async def test_get_document_without_credentials(supabase_service_without_credentials):
    """Test get_document returns None without Supabase"""
    document = await supabase_service_without_credentials.get_document("doc_123")

    assert document is None


@pytest.mark.asyncio
async def test_search_hybrid_empty_query(supabase_service_without_credentials):
    """Test search_hybrid with empty query"""
    documents = await supabase_service_without_credentials.search_hybrid(
        embedding=[0.1] * 1536,
        query="",
        k=5
    )

    assert isinstance(documents, list)


@pytest.mark.asyncio
async def test_document_structure(supabase_service_without_credentials):
    """Test returned documents have correct structure"""
    documents = await supabase_service_without_credentials.search_hybrid(
        embedding=[0.1] * 1536,
        query="test",
        k=1
    )

    doc = documents[0]
    assert hasattr(doc, "id")
    assert hasattr(doc, "content")
    assert hasattr(doc, "embedding")
    assert hasattr(doc, "metadata")
    assert hasattr(doc, "score")


@pytest.mark.asyncio
async def test_save_document_returns_string_id(supabase_service_without_credentials):
    """Test save_document returns string ID"""
    doc_id = await supabase_service_without_credentials.save_document(
        content="test",
        embedding=[0.1] * 1536,
        metadata={}
    )

    assert isinstance(doc_id, str)
    assert len(doc_id) > 0
