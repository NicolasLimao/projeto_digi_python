import pytest
from src.agents.formatter_agent import FormatterAgent


@pytest.fixture
def formatter():
    """FormatterAgent instance"""
    return FormatterAgent()


@pytest.mark.asyncio
async def test_format_orientacao_adds_bullets(formatter):
    """Test orientacao mode adds bullet points"""
    response = "First step\nSecond step\nThird step"

    formatted = await formatter.execute(response, "orientacao")

    assert "- " in formatted
    lines = formatted.split("\n")
    assert all(line.startswith("-") for line in lines if line.strip())


@pytest.mark.asyncio
async def test_format_orientacao_preserves_bullets(formatter):
    """Test orientacao mode preserves existing bullets"""
    response = "- Step 1\n- Step 2"

    formatted = await formatter.execute(response, "orientacao")

    assert formatted.count("-") >= 2


@pytest.mark.asyncio
async def test_format_resposta_cliente_removes_bullets(formatter):
    """Test resposta-cliente mode removes bullet points"""
    response = "- First point\n- Second point"

    formatted = await formatter.execute(response, "resposta-cliente")

    assert not formatted.startswith("- ")
    assert "First point" in formatted
    assert "Second point" in formatted


@pytest.mark.asyncio
async def test_format_resposta_cliente_plain_text(formatter):
    """Test resposta-cliente returns plain text"""
    response = "This is a message for the customer"

    formatted = await formatter.execute(response, "resposta-cliente")

    assert isinstance(formatted, str)
    assert formatted == response


@pytest.mark.asyncio
async def test_format_bug_adds_error_header(formatter):
    """Test bug mode adds ERROR header"""
    response = "Application crashed when saving"

    formatted = await formatter.execute(response, "bug")

    assert "BUG" in formatted or "ERRO" in formatted or "ERROR" in formatted


@pytest.mark.asyncio
async def test_format_bug_preserves_existing_header(formatter):
    """Test bug mode preserves existing ERROR header"""
    response = "ERRO: Application crashed"

    formatted = await formatter.execute(response, "bug")

    assert "ERRO:" in formatted


@pytest.mark.asyncio
async def test_format_empty_response(formatter):
    """Test formatter handles empty response"""
    formatted = await formatter.execute("", "orientacao")

    assert formatted == ""


@pytest.mark.asyncio
async def test_format_orientacao_numbered_list(formatter):
    """Test orientacao mode converts numbered lists"""
    response = "1. First\n2. Second\n3. Third"

    formatted = await formatter.execute(response, "orientacao")

    assert "- " in formatted


@pytest.mark.asyncio
async def test_format_resposta_cliente_multiple_paragraphs(formatter):
    """Test resposta-cliente preserves paragraph structure"""
    response = "First paragraph\n\nSecond paragraph\n\nThird paragraph"

    formatted = await formatter.execute(response, "resposta-cliente")

    assert "First paragraph" in formatted
    assert "Second paragraph" in formatted


@pytest.mark.asyncio
async def test_format_unknown_mode_returns_unchanged(formatter):
    """Test unknown mode returns response unchanged"""
    response = "Some response"

    formatted = await formatter.execute(response, "unknown_mode")

    assert formatted == response


@pytest.mark.asyncio
async def test_format_orientacao_removes_empty_lines(formatter):
    """Test orientacao mode removes empty lines"""
    response = "Step 1\n\nStep 2\n\n\nStep 3"

    formatted = await formatter.execute(response, "orientacao")

    assert "\n\n" not in formatted or formatted.count("\n\n") < response.count("\n\n")
