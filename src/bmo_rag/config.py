from pathlib import Path

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "bmo-rag"
    log_level: str = "INFO"
    llm_model: str = "gpt-5"
    openai_api_key: SecretStr | None = None
    embedding_model: str = "BAAI/bge-m3"
    vector_store: str = "qdrant"
    top_k: int = 5
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: SecretStr | None = None
    observability_db: Path = Path("data/observability/rag_observability.sqlite3")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )
