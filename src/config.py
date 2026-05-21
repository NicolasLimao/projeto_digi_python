from pydantic_settings import BaseSettings
from pydantic import ConfigDict
from typing import Optional


class Settings(BaseSettings):
    model_config = ConfigDict(env_file=".env", case_sensitive=False, extra="ignore")

    # OpenAI
    openai_api_key: str

    # Supabase
    supabase_url: str
    supabase_anon_key: str

    # Discord
    discord_bot_token: str
    discord_guild_id: int
    discord_channel_duvidas: int
    discord_channel_logs: int
    discord_channel_gaps: int

    # Mistral
    mistral_api_key: Optional[str] = None

    # Cohere
    cohere_api_key: Optional[str] = None

    # History service
    history_enabled: bool = True
    history_retention_days: int = 90

    # Discord DM
    discord_dm_enabled: bool = True

    # RAG API
    rag_api_url: str = "http://localhost:8000/api/rag/query"

    # App
    log_level: str = "INFO"
    environment: str = "development"
    score_threshold: float = 0.65
    max_chunks: int = 10


settings = Settings()
