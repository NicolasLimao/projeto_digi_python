from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
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

    # App
    log_level: str = "INFO"
    environment: str = "development"
    score_threshold: float = 0.65
    max_chunks: int = 10

    class Config:
        env_file = ".env"
        case_sensitive = False


settings = Settings()
