from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")
    
    PROJECT_NAME: str = "Calabi ML Server"
    DEBUG: bool = True
    VERSION: str = "1.0.0"
    PORT: int = 8000

    BACKEND_CORS_ORIGINS: list[str] = ["*"]

    # Canonicalization (OpenAI)
    CANONICALIZATION_ENABLED: bool = False
    OPENAI_API_KEY: str | None = None
    OPENAI_MODEL: str = "gpt-4o-mini"#"gpt-4.1-nano"#"gpt-4o-mini"
    OPENROUTER_API_KEY: str | None = None
    OPENROUTER_MODEL: str = "google/gemini-2.0-flash-001"#"google/gemini-2.0-flash-exp:free"

    # NER
    NER_MAX_MENTIONS: int = 12
    NER_MIN_TOKEN_LEN: int = 2
    NER_REQUEST_TIMEOUT_SEC: float = 8.0

    # GLiNER
    GLINER_MODEL: str = "gliner-community/gliner_large-v2.5"#"urchade/gliner_medium-v2.1"
    GLINER_THRESHOLD: float = 0.1

settings = Settings()