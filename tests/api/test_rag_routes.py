import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, MagicMock, patch
from src.api.rag_routes import router
from src.models.schemas import QueryRequest, QueryResponse
from fastapi import FastAPI


@pytest.fixture
def app():
    """FastAPI test app with RAG router"""
    app = FastAPI()
    app.include_router(router)
    return app


@pytest.fixture
def client(app):
    """Test client"""
    return TestClient(app)


@pytest.mark.asyncio
async def test_rag_query_endpoint_exists(client):
    """Test RAG query endpoint is available"""
    response = client.post(
        "/api/rag/query",
        params={"user_id": "test_user"},
        json={"query": "test query"}
    )

    assert response.status_code in [200, 422, 500]


def test_rag_query_missing_user_id(client):
    """Test query endpoint requires user_id"""
    response = client.post(
        "/api/rag/query",
        json={"query": "test query"}
    )

    assert response.status_code == 422


def test_rag_query_missing_query(client):
    """Test query endpoint requires query in body"""
    response = client.post(
        "/api/rag/query",
        params={"user_id": "test_user"},
        json={}
    )

    assert response.status_code == 422


def test_rag_query_empty_query(client):
    """Test query endpoint rejects empty query"""
    with patch("src.api.rag_routes.get_rag_pipeline") as mock_pipeline:
        pipeline_instance = AsyncMock()
        pipeline_instance.process = AsyncMock(
            return_value=QueryResponse(
                response="Error",
                mode="orientacao",
                score=0.0,
                chunks_used=0,
                processing_time_ms=100
            )
        )
        mock_pipeline.return_value = pipeline_instance

        response = client.post(
            "/api/rag/query",
            params={"user_id": "test_user"},
            json={"query": "   "}
        )

        assert response.status_code == 400


def test_rag_query_empty_user_id(client):
    """Test query endpoint rejects empty user_id"""
    with patch("src.api.rag_routes.get_rag_pipeline") as mock_pipeline:
        pipeline_instance = AsyncMock()
        pipeline_instance.process = AsyncMock(
            return_value=QueryResponse(
                response="Error",
                mode="orientacao",
                score=0.0,
                chunks_used=0,
                processing_time_ms=100
            )
        )
        mock_pipeline.return_value = pipeline_instance

        response = client.post(
            "/api/rag/query",
            params={"user_id": ""},
            json={"query": "test"}
        )

        assert response.status_code == 400


def test_rag_query_response_structure(client):
    """Test response has correct structure"""
    with patch("src.api.rag_routes.get_rag_pipeline") as mock_pipeline:
        pipeline_instance = AsyncMock()
        pipeline_instance.process = AsyncMock(
            return_value=QueryResponse(
                response="Test response",
                mode="orientacao",
                score=0.85,
                chunks_used=2,
                processing_time_ms=150
            )
        )
        mock_pipeline.return_value = pipeline_instance

        response = client.post(
            "/api/rag/query",
            params={"user_id": "test_user"},
            json={"query": "test query"}
        )

        assert response.status_code == 200
        data = response.json()
        assert "response" in data
        assert "mode" in data
        assert "score" in data
        assert "chunks_used" in data
        assert "processing_time_ms" in data


def test_rag_query_with_mode(client):
    """Test query endpoint accepts optional mode parameter"""
    with patch("src.api.rag_routes.get_rag_pipeline") as mock_pipeline:
        pipeline_instance = AsyncMock()
        pipeline_instance.process = AsyncMock(
            return_value=QueryResponse(
                response="Response",
                mode="resposta-cliente",
                score=0.80,
                chunks_used=1,
                processing_time_ms=100
            )
        )
        mock_pipeline.return_value = pipeline_instance

        response = client.post(
            "/api/rag/query",
            params={"user_id": "test_user"},
            json={"query": "test", "mode": "resposta-cliente"}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["mode"] == "resposta-cliente"


def test_rag_query_response_content_type(client):
    """Test response content type is JSON"""
    with patch("src.api.rag_routes.get_rag_pipeline") as mock_pipeline:
        pipeline_instance = AsyncMock()
        pipeline_instance.process = AsyncMock(
            return_value=QueryResponse(
                response="Test",
                mode="orientacao",
                score=0.0,
                chunks_used=0,
                processing_time_ms=0
            )
        )
        mock_pipeline.return_value = pipeline_instance

        response = client.post(
            "/api/rag/query",
            params={"user_id": "test_user"},
            json={"query": "test"}
        )

        assert response.headers["content-type"] == "application/json"


def test_rag_query_processing_time_positive(client):
    """Test processing_time_ms is always positive"""
    with patch("src.api.rag_routes.get_rag_pipeline") as mock_pipeline:
        pipeline_instance = AsyncMock()
        pipeline_instance.process = AsyncMock(
            return_value=QueryResponse(
                response="Response",
                mode="orientacao",
                score=0.5,
                chunks_used=1,
                processing_time_ms=250
            )
        )
        mock_pipeline.return_value = pipeline_instance

        response = client.post(
            "/api/rag/query",
            params={"user_id": "test_user"},
            json={"query": "test"}
        )

        assert response.status_code == 200
        assert response.json()["processing_time_ms"] >= 0


def test_rag_query_score_in_range(client):
    """Test score is between 0 and 1"""
    with patch("src.api.rag_routes.get_rag_pipeline") as mock_pipeline:
        pipeline_instance = AsyncMock()
        pipeline_instance.process = AsyncMock(
            return_value=QueryResponse(
                response="Response",
                mode="orientacao",
                score=0.75,
                chunks_used=2,
                processing_time_ms=100
            )
        )
        mock_pipeline.return_value = pipeline_instance

        response = client.post(
            "/api/rag/query",
            params={"user_id": "test_user"},
            json={"query": "test"}
        )

        data = response.json()
        assert 0 <= data["score"] <= 1
