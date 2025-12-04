from functools import lru_cache
from pydantic import BaseSettings


class Settings(BaseSettings):
    PROJECT_NAME: str = "Calabi ML Server"
    DEBUG: bool = True

    BACKEND_CORS_ORIGINS: list[str] = ["*"]

    # 전역 모델 dir
    MODEL_DIR: str = "./models"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache
def get_settings() -> Settings:
    return Settings()
