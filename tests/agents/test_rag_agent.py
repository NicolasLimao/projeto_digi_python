from unittest.mock import AsyncMock, MagicMock

import pytest

from src.agents.rag_agent import RAGAgent
from src.config import Settings
from src.models.schemas import Document


@pytest.fixture
def documents() -> list[Document]:
    return [
        Document(id="1", content="backup", embedding=[], metadata={"fonte": "manual"}, score=0.9),
        Document(id="2", content="restore", embedding=[], metadata={"fonte": "faq"}, score=0.7),
    ]


@pytest.fixture
def services(documents):
    openai = MagicMock()
    openai.get_embeddings = AsyncMock(return_value=[0.1] * 4)
    openai.rewrite_query = AsyncMock(side_effect=lambda query, history="": query)
    openai.rerank = AsyncMock(side_effect=lambda query, chunks, top_n=10: chunks[:top_n])
    openai.generate_response = AsyncMock(return_value="Resposta")
    openai.format_response = AsyncMock(side_effect=lambda response, mode: response.strip())
    supabase = MagicMock()
    supabase.search_hybrid = AsyncMock(return_value=documents)
    return openai, supabase


@pytest.fixture
def rag_agent(services):
    return RAGAgent(*services)


@pytest.mark.asyncio
async def test_retrieve_builds_metrics_and_sources(rag_agent):
    result = await rag_agent.retrieve("Como fazer backup?", k=10)
    assert result["chunks_used"] == 2
    assert result["score"] == pytest.approx(0.8)
    assert result["fontes"] == ["1", "2"]


@pytest.mark.asyncio
async def test_retrieve_expands_candidate_pool(rag_agent, services):
    await rag_agent.retrieve("teste", k=3)
    _, supabase = services
    assert supabase.search_hybrid.await_args.kwargs["k"] == 8


@pytest.mark.asyncio
async def test_retrieve_caps_final_results(rag_agent, services, documents):
    openai, _ = services
    await rag_agent.retrieve("teste", k=1)
    assert openai.rerank.await_args.kwargs["top_n"] == 1


@pytest.mark.asyncio
async def test_retrieve_uses_injected_limits_and_threshold(services):
    openai, supabase = services
    config = Settings(max_chunks=2, score_threshold=0.65)
    agent = RAGAgent(openai, supabase, config)
    await agent.retrieve("teste", k=10)
    assert openai.rerank.await_args.kwargs["top_n"] == 2
    assert supabase.search_hybrid.await_args.kwargs["score_threshold"] == 0.65


@pytest.mark.asyncio
async def test_rewrite_is_skipped_for_simple_query(rag_agent, services):
    openai, _ = services
    await rag_agent.retrieve("backup")
    openai.rewrite_query.assert_not_awaited()


@pytest.mark.asyncio
async def test_rewrite_is_used_for_followup(rag_agent, services):
    openai, _ = services
    await rag_agent.retrieve("e isso?", history_context="pergunta anterior")
    openai.rewrite_query.assert_awaited_once()


@pytest.mark.asyncio
async def test_generate_uses_documents_and_mode(rag_agent, services, documents):
    result = await rag_agent.generate("pergunta", documents, "bug", "histórico")
    openai, _ = services
    assert result == "Resposta"
    assert openai.generate_response.await_args.args[2] == "bug"


@pytest.mark.asyncio
async def test_generate_without_documents_returns_safe_message(rag_agent, services):
    result = await rag_agent.generate("desconhecido", [], "orientacao")
    openai, _ = services
    assert "não encontrei" in result
    openai.generate_response.assert_not_awaited()


@pytest.mark.asyncio
async def test_execute_returns_compatibility_payload(rag_agent):
    result = await rag_agent.execute("backup", mode="resposta-cliente")
    assert result["response"] == "Resposta"
    assert result["mode"] == "resposta-cliente"
    assert result["chunks_used"] == 2


@pytest.mark.asyncio
async def test_execute_hides_internal_exception(rag_agent, services):
    openai, _ = services
    openai.get_embeddings.side_effect = RuntimeError("secret internal detail")
    result = await rag_agent.execute("backup")
    assert "secret internal detail" not in result["response"]
    assert result["score"] == 0.0
