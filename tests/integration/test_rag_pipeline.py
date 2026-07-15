from unittest.mock import AsyncMock, MagicMock

import pytest

from src.models.schemas import Document, QueryResponse
from src.pipeline.rag_pipeline import RAGPipeline


@pytest.fixture
def components():
    classifier = MagicMock()
    classifier.execute = AsyncMock(return_value="orientacao")
    validator = MagicMock()
    validator.execute = AsyncMock(return_value={"dentro_do_escopo": True})
    document = Document(id="1", content="contexto", embedding=[], metadata={}, score=0.8)
    rag = MagicMock()
    rag.retrieve = AsyncMock(
        return_value={
            "documents": [document],
            "search_query": "consulta",
            "fontes": ["1"],
            "score": 0.8,
            "chunks_used": 1,
        }
    )
    rag.generate = AsyncMock(return_value=" resposta gerada ")
    formatter = MagicMock()
    formatter.execute = AsyncMock(return_value="resposta gerada")
    history = MagicMock()
    history.format_history_for_prompt = AsyncMock(return_value="")
    history.save_interaction = AsyncMock(return_value="history-1")
    return classifier, validator, rag, formatter, history


@pytest.fixture
def pipeline(components):
    return RAGPipeline(*components)


@pytest.mark.asyncio
async def test_process_returns_query_response(pipeline):
    result = await pipeline.process("pergunta", "user-1")
    assert isinstance(result, QueryResponse)
    assert result.response == "resposta gerada"
    assert result.interaction_id == "history-1"


@pytest.mark.asyncio
async def test_independent_steps_run(pipeline, components):
    classifier, validator, rag, _, history = components
    await pipeline.process("pergunta", "user-1")
    classifier.execute.assert_awaited_once()
    validator.execute.assert_awaited_once()
    rag.retrieve.assert_awaited_once()
    history.format_history_for_prompt.assert_awaited_once()


@pytest.mark.asyncio
async def test_explicit_mode_skips_classifier(pipeline, components):
    classifier, _, _, _, _ = components
    result = await pipeline.process("pergunta", "user-1", mode="bug")
    classifier.execute.assert_not_awaited()
    assert result.mode == "bug"


@pytest.mark.asyncio
async def test_result_uses_retrieval_metrics(pipeline):
    result = await pipeline.process("pergunta", "user-1")
    assert result.score == 0.8
    assert result.chunks_used == 1


@pytest.mark.asyncio
async def test_out_of_scope_returns_without_generation(pipeline, components):
    _, validator, rag, formatter, _ = components
    validator.execute.return_value = {"dentro_do_escopo": False, "motivo": "externo"}
    rag.retrieve.return_value = {
        "documents": [],
        "search_query": "q",
        "fontes": [],
        "score": 0.0,
        "chunks_used": 0,
    }
    result = await pipeline.process("bolo", "user-1")
    assert "fora do escopo" in result.response
    rag.generate.assert_not_awaited()
    formatter.execute.assert_not_awaited()


@pytest.mark.asyncio
async def test_strong_retrieval_overrides_false_negative(pipeline, components):
    _, validator, rag, _, _ = components
    validator.execute.return_value = {"dentro_do_escopo": False}
    rag.retrieve.return_value["chunks_used"] = 5
    rag.retrieve.return_value["score"] = 0.4
    result = await pipeline.process("webhook", "user-1")
    assert result.response == "resposta gerada"


@pytest.mark.asyncio
async def test_history_failure_does_not_replace_valid_response(pipeline, components):
    *_, history = components
    history.save_interaction.side_effect = RuntimeError("database unavailable")
    result = await pipeline.process("pergunta", "user-1")
    assert result.response == "resposta gerada"
    assert result.interaction_id is None


@pytest.mark.asyncio
async def test_pipeline_hides_internal_error(pipeline, components):
    _, _, rag, _, _ = components
    rag.retrieve.side_effect = RuntimeError("secret connection string")
    result = await pipeline.process("pergunta", "user-1")
    assert "secret connection string" not in result.response
    assert result.score == 0.0


@pytest.mark.asyncio
async def test_processing_time_is_non_negative(pipeline):
    result = await pipeline.process("pergunta", "user-1")
    assert result.processing_time_ms >= 0


@pytest.mark.asyncio
async def test_history_receives_retrieval_metadata(pipeline, components):
    *_, history = components
    await pipeline.process("pergunta", "user-1", canal="dm")
    kwargs = history.save_interaction.await_args.kwargs
    assert kwargs["pergunta_reescrita"] == "consulta"
    assert kwargs["fontes"] == ["1"]
    assert kwargs["canal"] == "dm"
