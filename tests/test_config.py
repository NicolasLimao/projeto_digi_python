import pytest
from pydantic import ValidationError

from src.config import Settings


def test_development_can_start_without_external_credentials():
    settings = Settings(environment="development")
    assert settings.openai_key is None
    assert settings.database_key is None


def test_empty_secrets_are_treated_as_unset():
    settings = Settings(openai_api_key="", api_auth_token="")
    assert settings.openai_api_key is None
    assert settings.api_auth_token is None


def test_production_requires_private_backend_credentials():
    with pytest.raises(ValidationError, match="Missing production configuration"):
        Settings(environment="production")


def test_production_configuration_uses_service_role():
    settings = Settings(
        environment="production",
        openai_api_key="openai-secret",
        supabase_url="https://project.supabase.co",
        supabase_service_role_key="service-secret",
        api_auth_token="api-secret",
    )
    assert settings.database_key == "service-secret"


@pytest.mark.parametrize(
    ("field", "value"),
    [("score_threshold", 1.1), ("max_chunks", 0), ("port", 70_000)],
)
def test_numeric_bounds_are_enforced(field: str, value):
    with pytest.raises(ValidationError):
        Settings(**{field: value})
