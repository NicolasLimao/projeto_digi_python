from functools import lru_cache
from typing import Annotated, Any, Self

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


def _csv_list(value: Any) -> Any:
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return value


class Settings(BaseSettings):
    """Application configuration loaded from environment variables.

    Development may start without external credentials so health checks and unit
    tests remain usable. Production deliberately fails fast unless every secret
    needed by the API is configured.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=False,
        extra="ignore",
    )

    openai_api_key: SecretStr | None = None
    openai_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"
    openai_timeout_seconds: float = Field(default=45.0, ge=5.0, le=120.0)
    openai_max_retries: int = Field(default=2, ge=0, le=5)

    supabase_url: str | None = None
    supabase_anon_key: SecretStr | None = None
    supabase_service_role_key: SecretStr | None = None

    mistral_api_key: SecretStr | None = None
    cohere_api_key: SecretStr | None = None

    api_auth_token: SecretStr | None = None
    cors_allowed_origins: Annotated[list[str], NoDecode] = Field(default_factory=list)
    trusted_hosts: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: ["localhost", "127.0.0.1", "testserver"]
    )

    ingest_allowed_hosts: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: ["cdn.discordapp.com", "media.discordapp.net"]
    )
    ingest_max_attachments: int = Field(default=5, ge=1, le=20)
    ingest_max_file_bytes: int = Field(default=20 * 1024 * 1024, ge=1024)
    ingest_max_total_chars: int = Field(default=2_000_000, ge=1_000)
    ingest_download_timeout_seconds: float = Field(default=30.0, ge=5.0, le=120.0)
    ingest_max_pdf_pages: int = Field(default=1_000, ge=1, le=5_000)

    history_enabled: bool = True
    history_retention_days: int = Field(default=90, ge=1, le=3_650)
    rag_api_url: str = "http://localhost:8000/api/rag/query"

    log_level: str = "INFO"
    environment: str = "development"
    score_threshold: float = Field(default=0.30, ge=0.0, le=1.0)
    max_chunks: int = Field(default=10, ge=1, le=30)
    host: str = "0.0.0.0"
    port: int = Field(default=8000, ge=1, le=65_535)

    sentry_dsn: SecretStr | None = None
    release_version: str | None = None
    sentry_traces_sample_rate: float = Field(default=0.1, ge=0.0, le=1.0)
    sentry_profiles_sample_rate: float = Field(default=0.0, ge=0.0, le=1.0)

    @field_validator("cors_allowed_origins", "trusted_hosts", "ingest_allowed_hosts", mode="before")
    @classmethod
    def parse_csv_lists(cls, value: Any) -> Any:
        return _csv_list(value)

    @field_validator("environment", mode="before")
    @classmethod
    def normalize_environment(cls, value: Any) -> str:
        return str(value or "development").strip().lower()

    @field_validator("log_level", mode="before")
    @classmethod
    def normalize_log_level(cls, value: Any) -> str:
        return str(value or "INFO").strip().upper()

    @field_validator(
        "openai_api_key",
        "supabase_url",
        "supabase_anon_key",
        "supabase_service_role_key",
        "mistral_api_key",
        "cohere_api_key",
        "api_auth_token",
        "sentry_dsn",
        mode="before",
    )
    @classmethod
    def empty_values_are_unset(cls, value: Any) -> Any:
        return None if isinstance(value, str) and not value.strip() else value

    @model_validator(mode="after")
    def validate_production_secrets(self) -> Self:
        if self.environment != "production":
            return self

        required = {
            "OPENAI_API_KEY": self.openai_api_key,
            "SUPABASE_URL": self.supabase_url,
            "SUPABASE_SERVICE_ROLE_KEY": self.supabase_service_role_key,
            "API_AUTH_TOKEN": self.api_auth_token,
        }
        missing = [name for name, value in required.items() if not value]
        if missing:
            raise ValueError("Missing production configuration: " + ", ".join(sorted(missing)))
        return self

    @property
    def database_key(self) -> str | None:
        secret = self.supabase_service_role_key or self.supabase_anon_key
        return secret.get_secret_value() if secret else None

    @property
    def openai_key(self) -> str | None:
        return self.openai_api_key.get_secret_value() if self.openai_api_key else None


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
